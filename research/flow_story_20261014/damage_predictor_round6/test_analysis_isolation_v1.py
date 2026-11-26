#!/usr/bin/env python3
"""Synthetic full 28-frame analyzer checks. Never reads real ROUND6 results.

NumPy CPU only. Synthetic check label mutations must not affect any deployable
decision/model; synthetic fit coarse-label mutation must not affect PCA.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import numpy as np
import predictor_models as pm
import analyze_predictors as analyzer
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MEASUREMENTS = []
BASE = "coarse_stage02"
ORDINARY = pm.BASE_FEATURES + ("psn1_activity", "psn1_spatial_neighbor_hamming")
REGION_COUNTS = {"all_valid": 1400, "B": 100, "L": 80, "B_and_L": 20,
                 "B_only": 80, "L_only_B_known": 50, "neither_B_known": 1170,
                 "B_unknown": 80, "voxel_nonzero": 1000, "voxel_zero": 400}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def metric(aee):
    return {"regions": {region: {"AEE": aee*(1+index*.1), "valid_pixels": count,
                                 "error_sum": aee*(1+index*.1)*count}
                        for index, (region, count) in enumerate(REGION_COUNTS.items())}}


def create_fixture(root, mode):
    root.mkdir()
    rng = np.random.RandomState(20261015)
    frames = []; groups = {"fit": [], "check": []}
    for group, nseq, nframe in (("fit", 10, 2), ("check", 2, 4)):
        for sequence_index in range(nseq):
            sequence = f"synthetic_{group}_sequence{sequence_index:02d}"
            for within in range(nframe):
                target = sequence+f"_frame{within:02d}"
                groups[group].append(target)
                cells = rng.normal(size=(1, 10, 14, 384)).astype(np.float32)
                cells += np.float32(sequence_index*.025+within*.01)
                truth = np.stack((2*cells[..., 0]-cells[..., 7]+.2,
                                  cells[..., 3]+.5*cells[..., 10]-.3), axis=-1).astype(np.float32)
                valid = rng.uniform(size=(1, 10, 14)) > .15
                truth[~valid] = np.nan
                if group == "check" and mode == "mutate_check_labels":
                    valid = ~valid
                    truth = np.full((1, 10, 14, 2), 123., dtype=np.float32)
                    truth[..., 1] = -321.
                    truth[~valid] = np.nan
                if group == "fit" and mode == "mutate_fit_head_labels":
                    truth = -2.5*truth+17.
                path = root/(target+".npz")
                np.savez(path, no_GT_cell_descriptor=cells, cell_GT_flow=truth, cell_GT_valid=valid)
                ordinary = []
                for index, tile in enumerate(pm.TILES):
                    vals = rng.uniform(.05, 1., size=7)
                    ordinary.append(dict(tile=list(tile), **{key: float(value) for key, value in zip(ORDINARY, vals)}))
                arms = {BASE: {"metrics": metric(3.+sequence_index*.01)}}
                branch = []
                for index, tile in enumerate(pm.TILES):
                    if group == "check":
                        # These two mutations deliberately reverse both references.
                        aee = 2.+.01*(index if mode == "mutate_check_labels" else 11-index)
                        mse = 12-index if mode == "mutate_check_labels" else index+1
                    else:
                        aee = 2.5 + .2*ordinary[index]["amplitude_abs_mean"]-.1*ordinary[index]["pooling_residual_mse"]
                        mse = .1 + .6*ordinary[index]["feature_rms"]+.03*index
                    arms[analyzer.arm(index)] = {"metrics": metric(aee)}
                    branch.append({"tile": list(tile), "MSE": float(mse)})
                frames.append({"target_id": target, "sequence": sequence, "group": group,
                               "arms": arms, "ordinary_scores": ordinary,
                               "offline_fine_coarse_branch_mse": branch,
                               "predecision_input_file": path.name, "predecision_input_file_sha256": sha(path)})
    raw = {"schema": "flow_damage_round6_singletons_v1", "status": "ROUND6_SINGLETON_COLLECTION_COMPLETE",
           "original_model_unchanged": True, "BN_buffers_unchanged": True,
           "full_forward_count": 366, "completed_targets": 28, "frames": frames,
           "scope": "Synthetic CPU fixture; no real predictions or source data"}
    result = root/"results.json"; result.write_text(json.dumps(raw, allow_nan=False))
    manifest = root/"manifest.json"; manifest.write_text(json.dumps({"groups": groups}))
    return result, manifest


def execute_fixture(root, mode):
    result, manifest = create_fixture(root, mode)
    output = root/"analysis"
    started = time.monotonic()
    process = subprocess.run([sys.executable, "-B", str(HERE/"analyze_predictors.py"),
                              "--results", str(result), "--results-sha256", sha(result),
                              "--manifest", str(manifest), "--output-dir", str(output)],
                             universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=120, env=os.environ.copy())
    if process.returncode:
        raise AssertionError(process.stdout+process.stderr)
    pred = json.loads((output/"predictions_before_check_evaluation.json").read_text())
    analysis = json.loads((output/"analysis.json").read_text())
    MEASUREMENTS.append({"synthetic_mode": mode, "exit_code": process.returncode,
                         "wall_seconds": time.monotonic()-started,
                         "prediction_artifact_sha256": sha(output/"predictions_before_check_evaluation.json"),
                         "selected_arms_sha256": digest(pred["selected_arms"]),
                         "fit_models_sha256": digest({"folds": pred["head_OOF_folds"], "final_heads": pred["final_heads"],
                                                      "selectors": pred["selector_models"]}),
                         "stdout": process.stdout.strip()})
    return pred, analysis


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="flow_round6_analysis_synthetic_")
        root = Path(cls.temp.name)
        cls.reference = execute_fixture(root/"reference", "reference")
        cls.check_mutation = execute_fixture(root/"check_labels_changed", "mutate_check_labels")
        cls.fit_mutation = execute_fixture(root/"fit_head_labels_changed", "mutate_fit_head_labels")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_synthetic_28frame_pipeline_and_predictions_before_check(self):
        pred, result = self.reference
        self.assertFalse(pred["check_GT_used_for_predictions"])
        self.assertEqual(len(pred["head_OOF_folds"]), 5)
        self.assertEqual(len(pred["fit_features"]), 20)
        self.assertEqual(len(pred["check_features"]), 8)
        self.assertEqual(len(pred["selected_arms"]), 16)
        self.assertNotIn("offline_branch_MSE", pred["selected_arms"])
        self.assertNotIn("posthoc_GT_best", pred["selected_arms"])
        self.assertEqual(len(result["check_summaries"]), 19)
        self.assertTrue(result["head_quality"]["check"]["not_dense_flow_AEE"])
        for fold in pred["head_OOF_folds"]:
            self.assertEqual(len(fold["train_ids"]), 16)
            self.assertEqual(len(fold["held_ids"]), 4)
            self.assertFalse(set(fold["train_ids"]) & set(fold["held_ids"]))
            self.assertTrue(all(not any(t.startswith(s+"_") for s in fold["held_sequences"]) for t in fold["train_ids"]))

    def test_all_check_label_changes_leave_fit_models_features_scores_decisions_exact(self):
        reference, _ = self.reference; changed, _ = self.check_mutation
        ignored = {"input_sha256"}
        for key in reference:
            if key not in ignored:
                self.assertEqual(reference[key], changed[key], key)
        self.assertNotEqual(reference["input_sha256"], changed["input_sha256"])
        MEASUREMENTS.append({"check_GT_and_branch_mutation": "all 8 check members: cell GT values/validity plus actual-endpoint metrics and branch MSE",
                             "all_16_deployable_policies_unchanged": True,
                             "all_fit_heads_PCA_selectors_features_scores_unchanged": True,
                             "only_prediction_provenance_results_SHA_changes": True})

    def test_mutated_check_labels_change_only_evaluation_and_offline_references(self):
        _, reference = self.reference; _, changed = self.check_mutation
        for key in ("offline_branch_MSE", "posthoc_GT_best"):
            self.assertNotEqual(reference["check_summaries"][key]["selected_arms"], changed["check_summaries"][key]["selected_arms"])
        self.assertNotEqual(reference["head_quality"]["check"], changed["head_quality"]["check"])
        self.assertEqual(reference["head_quality"]["fit_OOF"], changed["head_quality"]["fit_OOF"])
        for key in self.reference[0]["selected_arms"]:
            self.assertEqual(reference["check_summaries"][key]["selected_arms"], changed["check_summaries"][key]["selected_arms"])

    def test_fit_coarse_GT_changes_flow_head_but_never_PCA_or_nonflow_models(self):
        reference, _ = self.reference; changed, _ = self.fit_mutation
        self.assertNotEqual(reference["final_heads"]["flow"], changed["final_heads"]["flow"])
        self.assertNotEqual(reference["check_flow_prediction"], changed["check_flow_prediction"])
        self.assertEqual(reference["final_heads"]["PCA"], changed["final_heads"]["PCA"])
        self.assertEqual(reference["check_PCA_prediction"], changed["check_PCA_prediction"])
        for a, b in zip(reference["head_OOF_folds"], changed["head_OOF_folds"]):
            self.assertEqual(a["models"]["PCA"], b["models"]["PCA"])
            self.assertNotEqual(a["models"]["flow"], b["models"]["flow"])
        for name in reference["selector_models"]:
            if not name.startswith("flow11_"):
                self.assertEqual(reference["selector_models"][name], changed["selector_models"][name], name)
                self.assertEqual(reference["scores"][name], changed["scores"][name], name)
                self.assertEqual(reference["selected_arms"][name], changed["selected_arms"][name], name)
        MEASUREMENTS.append({"fit_cell_GT_mutation": "flow head and its outputs change, all fold/final PCA and nonflow predictors exact unchanged",
                             "PCA_has_no_GT_input": True})

    def test_fp32_folded_scalar_vector_prediction_and_feature_dtype(self):
        rng = np.random.RandomState(732)
        x = rng.normal(size=(50, 9)); test_x = rng.normal(size=(12, 9))
        for y in (rng.normal(size=50), rng.normal(size=(50, 2))):
            model = pm.ridge_fit(x, y, 1.)
            weight, bias = pm.folded(model)
            np.testing.assert_allclose(pm.predict(model, test_x), test_x @ weight+bias, atol=1e-14, rtol=1e-13)
            explicit_fp32 = test_x.astype(np.float32) @ weight.astype(np.float32)+np.asarray(bias, dtype=np.float32)
            actual = pm.predict_fp32(model, test_x)
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, explicit_fp32)
        ordinary = [dict(tile=list(t), **{key: .25 for key in ORDINARY}) for t in pm.TILES]
        families = pm.feature_families(ordinary, np.ones((10, 14, 384), np.float32),
                                       np.ones((10, 14, 2), np.float32), np.ones((10, 14, 2), np.float32))
        for name, width in (("B5", 5), ("rich37", 37), ("flow11", 11), ("PCA11", 11)):
            self.assertEqual(families[name].dtype, np.float32)
            self.assertEqual(families[name].shape, (12, width))
        self.assertNotIn("torch", sys.modules)


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources = [Path(__file__), HERE/"analyze_predictors.py", HERE/"predictor_models.py",
               HERE.parent/"damage_allocation_probe/analyze_interventions.py"]
    receipt = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
               "NumPy": np.__version__, "GPU_initialized": False, "torch_imported": "torch" in sys.modules,
               "full_model_run": False, "real_ROUND6_data_read": False, "synthetic_complete_pipeline_runs": 3,
               "source_sha256": {str(p): sha(p) for p in sources}, "measurements": MEASUREMENTS,
               "scope": "Implementer-side CPU isolation check on synthetic fit20/check8 only; not independent scientific review or measured experimental result",
               "errors": [str(x) for x in result.errors], "failures": [str(x) for x in result.failures]}
    (HERE/"analysis_cpu_results.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    return 0 if result.wasSuccessful() and not receipt["torch_imported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
