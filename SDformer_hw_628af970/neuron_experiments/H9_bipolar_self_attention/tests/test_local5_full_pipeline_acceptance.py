from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

import run_dsec_fullres_w15_h66d_local5_bb1e4_full_pipeline as pipeline
import run_dsec_fullres_w15_equal_plus10_convergence as convergence
import supervise_dsec_fullres_w15_h66d_local5_bb1e4 as supervisor


class Local5FullPipelineAcceptanceTest(unittest.TestCase):
    def test_deploy_summary_requires_dsec_fl_and_profile_equality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_epoch9.pth"
            checkpoint.write_bytes(b"checkpoint")
            profile = root / "profile.json"
            profile.write_text("{}\n", encoding="utf-8")
            metrics = {
                "AEE": 1.2,
                "AAE": 6.0,
                "AAE_Benchmark": 5.8,
                "DSEC_Fl": 8.0,
                "total_spikes_g": 80.0,
            }
            summary = {
                "best_epoch": 9,
                "checkpoint": str(checkpoint),
                "float": dict(metrics),
                "dyadic": dict(metrics),
                "hardware_order": dict(metrics),
                "dyadic_profile": str(profile),
                "hardware_profile": str(profile),
            }
            (root / "deploy_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with (
                patch.object(pipeline, "ROOT", root),
                patch.object(pipeline, "best_epoch", return_value=9),
                patch.object(pipeline, "validate_eval_profile_contract"),
                patch.object(pipeline, "parse_profile", return_value=metrics),
                patch.object(pipeline, "record"),
            ):
                pipeline.validate_deploy_summary_contract(checkpoint)
                summary["hardware_order"].pop("DSEC_Fl")
                (root / "deploy_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "DSEC|metrics"):
                    pipeline.validate_deploy_summary_contract(checkpoint)

    def test_supervisor_marker_requires_strict_artifact_revalidation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = root / "status.log"
            status.write_text(supervisor.FINAL_MARKER + "\n", encoding="utf-8")
            with (
                patch.object(supervisor, "ROOT", root),
                patch.object(supervisor.pipeline, "validate_checkpoint_contract"),
                patch.object(supervisor.pipeline, "EVAL_EPOCHS", (9, 14)),
                patch.object(supervisor.pipeline, "validate_eval_profile_contract"),
                patch.object(supervisor.pipeline, "best_epoch", return_value=14),
                patch.object(supervisor.pipeline, "validate_deploy_summary_contract"),
                patch.object(supervisor.pipeline, "validate_profile_acceptance"),
            ):
                self.assertTrue(supervisor.complete())
                supervisor.pipeline.validate_profile_acceptance.side_effect = RuntimeError(
                    "stale acceptance"
                )
                self.assertFalse(supervisor.complete())

    def test_equal_plus10_criterion_matches_right_censoring_logic(self) -> None:
        self.assertIn("largest observed budget", convergence.CONVERGENCE_CRITERION)
        self.assertIn("descriptive only", convergence.CONVERGENCE_CRITERION)
        self.assertNotIn(">1%", convergence.CONVERGENCE_CRITERION)

    def test_equal_plus10_wait_requires_current_local5_rank1_rtl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            ranking = run / "profile_ranking_valid825.md"
            ranking.write_text(
                "| rank | epoch | AEE |\n|---:|---:|---:|\n| 1 | 29 | 1.2 |\n",
                encoding="utf-8",
            )
            checkpoint = run / "checkpoint_epoch29.pth"
            checkpoint.write_bytes(b"checkpoint")
            config = root / "hardware.yml"
            config.write_text("fixture: true\n", encoding="utf-8")
            report = root / "scope.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "evidence_scope": (
                            "checkpoint_bound_component_rtl_exact_not_full_network"
                        ),
                        "checkpoint_identity": {
                            "best_epoch": 29,
                            "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_sha256": convergence.sha256(checkpoint),
                            "config": str(config.resolve()),
                            "config_sha256": convergence.sha256(config),
                        },
                        "score_shiftmax": {"status": "PASS"},
                        "projection": {
                            "weight_mode": (
                                "checkpoint_theta_folded_dyadic_int8_head_slice"
                            ),
                            "verification": {
                                "checkpoint_weight_binding": "PASS",
                                "random_sva": "PASS",
                                "verilator_lint": "PASS",
                                "yosys_check": "PASS",
                            },
                        },
                        "atlif_temporal_matrix": {"status": "PASS"},
                    }
                ),
                encoding="utf-8",
            )
            status = root / "status.log"
            status.write_text(convergence.LOCAL5_RTL_COMPLETE + "\n", encoding="utf-8")
            with (
                patch.object(convergence, "LOCAL5_RUN", run),
                patch.object(convergence, "LOCAL5_RANKING", ranking),
                patch.object(convergence, "LOCAL5_HARDWARE_CONFIG", config),
                patch.object(convergence, "LOCAL5_RTL", report),
                patch.object(convergence, "LOCAL5_RTL_STATUS", status),
            ):
                self.assertTrue(convergence.local5_rtl_evidence_complete())
                value = json.loads(report.read_text(encoding="utf-8"))
                value["projection"]["weight_mode"] = (
                    "checkpoint_dyadic_int8_head_slice"
                )
                report.write_text(json.dumps(value), encoding="utf-8")
                self.assertFalse(convergence.local5_rtl_evidence_complete())
                value["projection"]["weight_mode"] = (
                    "checkpoint_theta_folded_dyadic_int8_head_slice"
                )
                value["checkpoint_identity"]["checkpoint_sha256"] = "0" * 64
                report.write_text(json.dumps(value), encoding="utf-8")
                self.assertFalse(convergence.local5_rtl_evidence_complete())

    def test_standard_profile_reuse_requires_complete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profile_ranking_valid825.md").write_text(
                "| rank | epoch |\n", encoding="utf-8"
            )
            with (
                patch.object(pipeline, "ROOT", root),
                patch.object(pipeline, "validate_eval_profile_contract") as validate,
            ):
                self.assertTrue(pipeline.standard_profiles_reusable())
                self.assertEqual(validate.call_count, len(pipeline.EVAL_EPOCHS))
                validate.reset_mock()
                validate.side_effect = FileNotFoundError("stale")
                self.assertFalse(pipeline.standard_profiles_reusable())

    def test_equal_profile_reuse_requires_complete_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profile_ranking_valid825.md").write_text(
                "| rank | epoch |\n", encoding="utf-8"
            )
            candidate = convergence.Candidate(
                name="fixture",
                config=root / "config.yml",
                source_model=root / "model.pth",
                source_state=root / "state.pth",
                root=root,
                source_label=0,
                final_label=1,
                eval_labels=(0, 1),
                expected_overlay_keys=0,
                expected_atlif=0,
                expected_shiftmax=0,
            )
            with patch.object(convergence, "validate_eval_profiles") as validate:
                self.assertTrue(convergence.eval_profiles_reusable(candidate))
                validate.side_effect = RuntimeError("stale")
                self.assertFalse(convergence.eval_profiles_reusable(candidate))

    def test_checkpoint_contract_requires_all_eval_models_and_resume_states(self) -> None:
        self.assertEqual(pipeline.EVAL_EPOCHS, (9, 14, 19, 24, 29))
        self.assertEqual(pipeline.RESUME_EPOCHS, (9, 19, 29))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            required = [
                *(root / f"checkpoint_epoch{epoch}.pth" for epoch in pipeline.EVAL_EPOCHS),
                *(
                    root / f"checkpoint_epoch{epoch}_state_dict.pth"
                    for epoch in pipeline.RESUME_EPOCHS
                ),
            ]
            for path in required:
                path.write_bytes(b"fixture")
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "record"):
                pipeline.validate_checkpoint_contract()
                for path in required:
                    with self.subTest(missing=path.name):
                        path.unlink()
                        with self.assertRaisesRegex(RuntimeError, path.name):
                            pipeline.validate_checkpoint_contract()
                        path.write_bytes(b"fixture")

    def test_equal_plus10_wait_requires_all_h67_ep30_component_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            config = root / "config.yml"
            profile = root / "profile.json"
            trace = root / "trace.json"
            audit = root / "audit.json"
            score = root / "score.json"
            atlif = root / "atlif.json"
            projection = root / "projection.json"
            status = root / "status.log"
            checkpoint.write_bytes(b"checkpoint")
            config.write_text("fixture: true\n", encoding="utf-8")
            checkpoint_sha = convergence.sha256(checkpoint)
            config_sha = convergence.sha256(config)
            records = []
            for index in range(12):
                payload = root / f"trace_{index}.npz"
                payload.write_bytes(f"trace-{index}".encode())
                records.append(
                    {
                        "file": str(payload),
                        "sha256": convergence.sha256(payload),
                        "temporal_tokens": 450,
                    }
                )
            profile.write_text(
                json.dumps(
                    {
                        "samples": 100,
                        "bit_trace_records": 12,
                        "artifact_identity": {
                            "checkpoint_sha256": checkpoint_sha,
                            "config_sha256": config_sha,
                        },
                        "eval_protocol": {
                            "resolution": [480, 640],
                            "crop": None,
                            "window_size": [2, 15, 15],
                            "tokens_per_window": 450,
                        },
                        "module_counts": {
                            "ATLIFTernaryPSN": 105,
                            "ShiftmaxAttention": 12,
                        },
                    }
                ),
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps(
                    {
                        "run_context": {
                            "artifact_identity": {
                                "checkpoint_sha256": checkpoint_sha,
                                "config_sha256": config_sha,
                            }
                        },
                        "records": records,
                    }
                ),
                encoding="utf-8",
            )
            audit.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "source_manifest": str(trace),
                        "coverage": {
                            "four_stage_complete": True,
                            "stages": [0, 1, 2, 3],
                        },
                        "records": [{"sha256_ok": True} for _ in range(12)],
                    }
                ),
                encoding="utf-8",
            )
            score.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "scope": "checkpoint_bound_component_rtl_exact",
                        "run_context": {
                            "artifact_identity": {
                                "checkpoint_sha256": checkpoint_sha,
                                "config_sha256": config_sha,
                            }
                        },
                        "source_trace_manifest": str(trace),
                        "source_trace_manifest_sha256": convergence.sha256(trace),
                    }
                ),
                encoding="utf-8",
            )
            atlif.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "checkpoint_identity": {
                            "checkpoint_sha256": checkpoint_sha,
                            "config_sha256": config_sha,
                        },
                    }
                ),
                encoding="utf-8",
            )
            projection.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "scope": "projection_component_rtl_exact",
                        "checkpoint_identity": {
                            "checkpoint_sha256": checkpoint_sha,
                            "config_sha256": config_sha,
                        },
                        "record_count": 12,
                        "required_stage_coverage": [0, 1, 2, 3],
                        "temporal_tokens": 450,
                        "token_id_width": 9,
                    }
                ),
                encoding="utf-8",
            )
            status.write_text(convergence.H67_EP30_COMPLETE + "\n", encoding="utf-8")
            candidate = convergence.Candidate(
                **{
                    **convergence.H67.__dict__,
                    "source_model": checkpoint,
                }
            )
            with (
                patch.object(convergence, "H67", candidate),
                patch.object(convergence, "H67_EP30_CONFIG", config),
                patch.object(convergence, "H67_EP30_PROFILE", profile),
                patch.object(convergence, "H67_EP30_TRACE", trace),
                patch.object(convergence, "H67_EP30_AUDIT", audit),
                patch.object(convergence, "H67_EP30_RTL", score),
                patch.object(convergence, "H67_EP30_ATLIF_RTL", atlif),
                patch.object(convergence, "H67_EP30_PROJECTION_RTL", projection),
                patch.object(convergence, "H67_EP30_STATUS", status),
            ):
                self.assertTrue(convergence.h67_ep30_evidence_complete())
                trace_value = json.loads(trace.read_text(encoding="utf-8"))
                trace_value["run_context"]["artifact_identity"]["config_sha256"] = "0" * 64
                trace.write_text(json.dumps(trace_value), encoding="utf-8")
                self.assertFalse(convergence.h67_ep30_evidence_complete())
                trace_value["run_context"]["artifact_identity"]["config_sha256"] = config_sha
                trace.write_text(json.dumps(trace_value), encoding="utf-8")
                score_value = json.loads(score.read_text(encoding="utf-8"))
                score_value["source_trace_manifest_sha256"] = convergence.sha256(trace)
                score.write_text(json.dumps(score_value), encoding="utf-8")
                self.assertTrue(convergence.h67_ep30_evidence_complete())
                atlif.write_text('{"status": "PASS"}\n', encoding="utf-8")
                self.assertFalse(convergence.h67_ep30_evidence_complete())

    def test_profile_acceptance_wait_releases_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            acceptance = Path(temporary) / "acceptance.json"
            acceptance.write_text("{}\n", encoding="utf-8")
            with patch.object(pipeline, "PROFILE_ACCEPTANCE", acceptance):
                pipeline.wait_for_profile_acceptance(poll_seconds=0, timeout_hours=0)

    def test_profile_acceptance_wait_times_out_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            acceptance = Path(temporary) / "acceptance.json"
            with patch.object(pipeline, "PROFILE_ACCEPTANCE", acceptance):
                with self.assertRaises(TimeoutError):
                    pipeline.wait_for_profile_acceptance(
                        poll_seconds=0, timeout_hours=0
                    )

    def test_profile_acceptance_wait_rejects_stale_until_strict_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            acceptance = Path(temporary) / "acceptance.json"
            checkpoint = Path(temporary) / "checkpoint.pth"
            acceptance.write_text("{}\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            with (
                patch.object(pipeline, "PROFILE_ACCEPTANCE", acceptance),
                patch.object(
                    pipeline,
                    "validate_profile_acceptance",
                    side_effect=[RuntimeError("stale"), None],
                ) as validate,
                patch.object(pipeline, "record"),
            ):
                pipeline.wait_for_profile_acceptance(
                    checkpoint, poll_seconds=0, timeout_hours=1
                )
                self.assertEqual(validate.call_count, 2)

    def _profile(
        self, checkpoint: Path, validation_file: Path, config: Path
    ) -> dict[str, object]:
        validation_file.write_text("fixture\n", encoding="utf-8")
        return {
            "samples": 825,
            "eval_protocol": {
                "resolution": [480, 640],
                "crop": None,
                "window_size": [2, 15, 15],
                "bn_policy": "no_running",
                "eval_batch_size": 1,
            },
            "metric_contract": {
                "AAE": "legacy_2d_direction_angle_degrees_between_uv",
                "AAE_Benchmark": (
                    "middlebury_barron_3d_angle_degrees_between_normalized_uv1"
                ),
                "DSEC_Fl": (
                    "percentage_epe_gt_3px_and_gt_5pct_of_ground_truth_flow_magnitude"
                ),
                "aggregation": (
                    "masked_mean_per_frame_then_equal_mean_over_validation_frames"
                ),
                "population": "local_DSEC_valid_file_list_not_official_hidden_test",
            },
            "metrics": {
                "AEE": 1.2,
                "AAE": 6.0,
                "AAE_Benchmark": 5.8,
                "DSEC_Fl": 8.0,
            },
            "metric_aggregation_audit": {
                "schema": "flow_metric_aggregation_audit_v1",
                "frame_count": 825,
                "valid_pixels": 1000.0,
                "sequence_count": 18,
                "frame_equal_mean": {
                    "AEE": 1.2, "AAE": 6.0, "AAE_Benchmark": 5.8, "DSEC_Fl": 8.0
                },
                "pixel_global_mean": {
                    "AEE": 1.1, "AAE": 5.9, "AAE_Benchmark": 5.7, "DSEC_Fl": 7.8
                },
                "sequence_balanced_mean": {
                    "AEE": 1.3, "AAE": 6.1, "AAE_Benchmark": 5.9, "DSEC_Fl": 8.2
                },
                "per_sequence": {
                    str(index): {"frame_count": 46 if index < 17 else 43}
                    for index in range(18)
                },
            },
            "validation_file_list": {
                "path": str(validation_file),
                "sha256": convergence.sha256(validation_file),
            },
            "checkpoint_load_audit": {
                "checkpoint_overlay_keys": 210,
                "model_overlay_keys": 210,
                "missing_count": 0,
                "unexpected_count": 0,
            },
            "module_counts": {
                "ATLIFTernaryPSN": 105,
                "ShiftmaxAttention": 12,
            },
            "artifact_identity": {
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": "checkpoint-sha",
                "config_path": str(config.resolve()),
                "config_sha256": convergence.sha256(config),
            },
        }

    def test_eval_profile_contract_accepts_and_rejects_count_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.pth"
            config = root / "config.yml"
            profile = root / "spike_profile.json"
            checkpoint.write_bytes(b"checkpoint")
            config.write_text("experiment: fixture\n", encoding="utf-8")
            value = self._profile(checkpoint, root / "valid.csv", config)
            profile.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(
                pipeline, "checkpoint_sha256", return_value="checkpoint-sha"
            ):
                pipeline.validate_eval_profile_contract(profile, checkpoint, config)
                value["module_counts"]["ShiftmaxAttention"] = 11
                profile.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Shiftmax12"):
                    pipeline.validate_eval_profile_contract(profile, checkpoint, config)
                value["module_counts"]["ShiftmaxAttention"] = 12
                value["metric_contract"]["population"] = "official_hidden_test"
                profile.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "local validation population"):
                    pipeline.validate_eval_profile_contract(profile, checkpoint, config)
                value["metric_contract"]["population"] = (
                    "local_DSEC_valid_file_list_not_official_hidden_test"
                )
                value["samples"] = 824
                profile.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "samples825"):
                    pipeline.validate_eval_profile_contract(profile, checkpoint, config)
                value["samples"] = 825
                value["artifact_identity"]["config_sha256"] = "stale-config-sha"
                profile.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "config SHA256"):
                    pipeline.validate_eval_profile_contract(profile, checkpoint, config)

    def test_post_g0_acceptance_binds_checkpoint_and_threshold_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.pth"
            config = root / "hardware.yml"
            manifest = root / "manifest.json"
            identity = root / "identity.json"
            acceptance = root / "acceptance.json"
            checkpoint.write_bytes(b"checkpoint")
            config.write_text("hardware: true\n", encoding="utf-8")
            identity.write_text(
                json.dumps(
                    {
                        "schema": "local5_post_g0_run_identity_v3",
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": pipeline.file_sha256(checkpoint),
                        "config": str(config.resolve()),
                        "config_sha256": pipeline.file_sha256(config),
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": pipeline.file_sha256(checkpoint),
                        "run_identity_file_sha256": pipeline.file_sha256(identity),
                    }
                ),
                encoding="utf-8",
            )
            required_checks = {
                name: True
                for name in (
                    "loader_provenance",
                    "formal_qualification",
                    "relation_rtl_binding",
                    "descriptor_geometry",
                    "replay_binding",
                    "descriptor_report_binding",
                    "reports_recomputed_equal",
                    "source_software_binding",
                    "release_receipt_binding",
                    "checkpoint_projection_weight_binding",
                    "checkpoint_projection_payload_recomputed",
                    "checkpoint_projection_topology_abi",
                    "threshold_training_deployment_semantics",
                )
            }
            value = {
                "schema": "local5_post_g0_acceptance_v1",
                "accepted": True,
                "samples": 100,
                "blocks": 12,
                "manifest": str(manifest),
                "manifest_sha256": pipeline.file_sha256(manifest),
                "run_identity": str(identity),
                "run_identity_sha256": pipeline.file_sha256(identity),
                "threshold_training_semantics": {
                    "threshold_modes": ["official_atlif"],
                    "homeostatic_freeze_after_step": 1224,
                    "optimizer_gradient_freeze_enabled": False,
                    "inference_threshold_source": "checkpoint_static_parameter",
                },
                "checks": required_checks,
            }
            acceptance.write_text(json.dumps(value), encoding="utf-8")
            with (
                patch.object(pipeline, "PROFILE_ACCEPTANCE", acceptance),
                patch.object(pipeline, "HARDWARE_CONFIG", config),
                patch.object(pipeline, "record"),
            ):
                pipeline.validate_profile_acceptance(checkpoint)
                value["samples"] = 99
                acceptance.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "samples100"):
                    pipeline.validate_profile_acceptance(checkpoint)
                value["samples"] = 100
                acceptance.write_text(json.dumps(value), encoding="utf-8")
                checkpoint.write_bytes(b"checkpoint drift")
                with self.assertRaisesRegex(
                    RuntimeError, "identity checkpoint SHA|manifest checkpoint SHA"
                ):
                    pipeline.validate_profile_acceptance(checkpoint)

    def test_equal_budget_profile_contract_accepts_and_rejects_overlay_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.yml"
            checkpoint = root / "checkpoint_epoch1.pth"
            profile = root / "standard_valid825/epoch1/spike_profile.json"
            config.write_text("test: true\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            profile.parent.mkdir(parents=True)
            candidate = convergence.Candidate(
                name="fixture",
                config=config,
                source_model=checkpoint,
                source_state=root / "state.pth",
                root=root,
                source_label=0,
                final_label=1,
                eval_labels=(1,),
                expected_overlay_keys=210,
                expected_atlif=105,
                expected_shiftmax=12,
            )
            value = self._profile(checkpoint, root / "valid.csv", config)
            value["artifact_identity"]["config_sha256"] = convergence.sha256(
                config
            )
            value["artifact_identity"]["checkpoint_sha256"] = convergence.sha256(
                checkpoint
            )
            profile.write_text(json.dumps(value), encoding="utf-8")
            with patch.object(convergence, "record"):
                convergence.validate_eval_profiles(candidate)
                value["checkpoint_load_audit"]["checkpoint_overlay_keys"] = 209
                profile.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "overlay count"):
                    convergence.validate_eval_profiles(candidate)

                value["checkpoint_load_audit"]["checkpoint_overlay_keys"] = 210
                value["metric_aggregation_audit"]["frame_count"] = 824
                profile.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "aggregation frames825"):
                    convergence.validate_eval_profiles(candidate)

                value["metric_aggregation_audit"]["frame_count"] = 825
                del value["metric_aggregation_audit"]["pixel_global_mean"]["AAE"]
                profile.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "aggregation modes complete"):
                    convergence.validate_eval_profiles(candidate)

                value["metric_aggregation_audit"]["pixel_global_mean"]["AAE"] = 5.9
                value["artifact_identity"]["config_path"] = str(
                    (root / "stale-config.yml").resolve()
                )
                profile.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "config path"):
                    convergence.validate_eval_profiles(candidate)

    def test_convergence_summary_separates_aee_angle_and_spike_trends(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def candidate(name: str) -> convergence.Candidate:
                run = root / name
                for label, metrics in zip(
                    (30, 35, 40),
                    ((1.50, 6.00, 5.80, 80.0), (1.45, 5.99, 5.79, 79.0), (1.42, 5.98, 5.78, 78.0)),
                    strict=True,
                ):
                    profile = run / f"standard_valid825/epoch{label}/spike_profile.json"
                    profile.parent.mkdir(parents=True, exist_ok=True)
                    profile.write_text(
                        json.dumps(
                            {
                                "metrics": {
                                    "AEE": metrics[0],
                                    "AAE": metrics[1],
                                    "AAE_Benchmark": metrics[2],
                                    "DSEC_Fl": 8.0,
                                },
                                "total_spikes": metrics[3] * 1e9,
                            }
                        ),
                        encoding="utf-8",
                    )
                return convergence.Candidate(
                    name=name,
                    config=root / f"{name}.yml",
                    source_model=root / "unused.pth",
                    source_state=root / "unused-state.pth",
                    root=run,
                    source_label=30,
                    final_label=40,
                    eval_labels=(30, 35, 40),
                    expected_overlay_keys=0,
                    expected_atlif=0,
                    expected_shiftmax=0,
                )

            local5 = candidate("Local5-fixture")
            h67 = candidate("H67-fixture")
            nb0 = candidate("NB0-fixture")
            with (
                patch.object(convergence, "LOCAL5", local5),
                patch.object(convergence, "H67", h67),
                patch.object(convergence, "NB0", nb0),
                patch.object(convergence, "CANDIDATES", (local5, h67, nb0)),
                patch.object(convergence, "SUMMARY_JSON", root / "summary.json"),
                patch.object(convergence, "SUMMARY_MD", root / "summary.md"),
                patch.object(convergence, "record"),
            ):
                result = convergence.write_convergence_summary()
            row = result["candidates"]["H67-fixture"]
            self.assertEqual(row["decision"], "not_plateaued")
            self.assertEqual(row["angle_decision"], "angle_plateaued")
            self.assertLess(row["spikes_last10_change_pct"], 0.0)
            self.assertIn("aae2d_last5_improvement_pct", row)
            self.assertIn("ae3d_last10_improvement_pct", row)


if __name__ == "__main__":
    unittest.main()
