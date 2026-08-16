from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from unittest.mock import patch

import neuron_experiments.H9_bipolar_self_attention.entrypoints.audit_date_algorithm_closure_20260805 as closure

from neuron_experiments.H9_bipolar_self_attention.entrypoints.audit_date_algorithm_closure_20260805 import (
    parse_ranking,
    sha256,
    validate_local_acceptance,
    validate_local_paired_states,
    validate_profile,
)


class DateAlgorithmClosureAuditTest(unittest.TestCase):
    def test_closure_auditor_source_is_hashable(self) -> None:
        auditor = Path(closure.__file__).resolve()
        self.assertTrue(auditor.is_file())
        self.assertEqual(len(sha256(auditor)), 64)

    def test_aae_gap_receipt_is_fail_closed(self) -> None:
        valid = {
            "status": "PASS_LOCAL_DIAGNOSIS_OFFICIAL_TEST_REPRODUCTION_UNAVAILABLE",
            "scope": "local diagnosis; not an official hidden-test reproduction",
            "metric_receipt_checks": {"source_sha": True},
            "head_to_head_checks": {"profile_sha": True},
            "diagnosis": {
                "formula_bug": False,
                "NB0_AEE_undertraining_plausible": True,
                "NB0_angle_gap_explained_by_undertraining_alone": False,
            },
            "late_trends": {"NB0": {}, "H67": {}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "aae_gap.json"
            receipt.write_text(json.dumps(valid), encoding="utf-8")
            with patch.object(closure, "AAE_GAP_RECEIPT", receipt):
                result = closure.validate_aae_gap_receipt()
                self.assertTrue(all(result["checks"].values()))
                valid["diagnosis"]["formula_bug"] = True
                receipt.write_text(json.dumps(valid), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "AAE gap receipt failed"):
                    closure.validate_aae_gap_receipt()

    def test_local_projection_requires_theta_folded_production_mode(self) -> None:
        closure.validate_local_projection_weight_mode(
            {"weight_mode": "checkpoint_theta_folded_dyadic_int8_head_slice"}
        )
        with self.assertRaisesRegex(RuntimeError, "theta-folded production mode"):
            closure.validate_local_projection_weight_mode(
                {"weight_mode": "checkpoint_dyadic_int8_head_slice"}
            )

    def test_algorithm_target_selects_only_qualifying_candidate(self) -> None:
        result = closure.select_qualifying_mainline(
            local5={"AEE": 1.60, "total_spikes_g": 70.0},
            h67={"AEE": 1.34, "total_spikes_g": 81.0},
            nb0={"AEE": 1.45, "total_spikes_g": 126.0},
        )
        self.assertEqual(result["selected_mainline"], "H67")
        self.assertTrue(result["candidates"]["H67"]["qualifies"])
        self.assertFalse(result["candidates"]["Local5"]["qualifies"])
        with self.assertRaisesRegex(RuntimeError, "no convergence-eligible H67/Local5"):
            closure.select_qualifying_mainline(
                local5={"AEE": 1.60, "total_spikes_g": 120.0},
                h67={"AEE": 1.70, "total_spikes_g": 80.0},
                nb0={"AEE": 1.45, "total_spikes_g": 126.0},
            )

    def test_mainline_selection_excludes_nonconverged_boundary_candidate(self) -> None:
        result = closure.select_qualifying_mainline(
            local5={"AEE": 1.20, "total_spikes_g": 70.0},
            h67={"AEE": 1.34, "total_spikes_g": 81.0},
            nb0={"AEE": 1.45, "total_spikes_g": 126.0},
            convergence_eligible={"Local5": False, "H67": True},
        )
        self.assertEqual(result["selected_mainline"], "H67")
        self.assertFalse(result["candidates"]["Local5"]["qualifies"])

    def test_local5_boundary_convergence_gate(self) -> None:
        profiles = {
            "24": {"AEE": 1.50},
            "29": {"AEE": 1.47},
        }
        boundary = closure.classify_local5_convergence(profiles, 29)
        self.assertEqual(boundary["decision"], "not_plateaued")
        self.assertFalse(boundary["final_mainline_eligible"])
        interior = closure.classify_local5_convergence(profiles, 24)
        self.assertTrue(interior["final_mainline_eligible"])

    def test_local5_small_slope_boundary_remains_right_censored(self) -> None:
        profiles = {
            "24": {"AEE": 1.5000},
            "29": {"AEE": 1.4990},
        }
        boundary = closure.classify_local5_convergence(profiles, 29)
        self.assertLess(boundary["aee_last5_improvement_pct"], 1.0)
        self.assertEqual(boundary["decision"], "not_plateaued")
        self.assertFalse(boundary["final_mainline_eligible"])

    def test_ordered_source_manifest_binds_checkpoint_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "ordered.json"
            source.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": "a" * 64,
                        "config_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            vector = {
                "source_manifest": str(source),
                "source_manifest_sha256": sha256(source),
            }
            self.assertEqual(
                closure.ordered_manifest_identity(vector),
                {
                    "checkpoint_sha256": "a" * 64,
                    "config_sha256": "b" * 64,
                },
            )
            value = json.loads(source.read_text(encoding="utf-8"))
            value["config_sha256"] = "short"
            source.write_text(json.dumps(value), encoding="utf-8")
            vector["source_manifest_sha256"] = sha256(source)
            with self.assertRaisesRegex(RuntimeError, "checkpoint/config"):
                closure.ordered_manifest_identity(vector)

    def test_convergence_profiles_carries_rank1_aggregation(self) -> None:
        summary = {
            "schema": "dsec_fullres_equal_plus10_convergence_v1",
            "candidates": {},
        }
        labels_by_name = {"H67": (30, 35, 40), "NB0": (29, 34, 39)}
        for name, labels in labels_by_name.items():
            summary["candidates"][name] = {
                "rank1_budget": 30,
                "rank1_checkpoint_label": labels[0],
                "decision": "operationally_plateaued_or_overfit",
                "angle_decision": "angle_plateaued",
                "aee_last5_improvement_pct": 0.0,
                "aee_last10_improvement_pct": 0.0,
                "aae2d_last5_improvement_pct": 0.0,
                "aae2d_last10_improvement_pct": 0.0,
                "ae3d_last5_improvement_pct": 0.0,
                "ae3d_last10_improvement_pct": 0.0,
                "spikes_last5_change_pct": 0.0,
                "spikes_last10_change_pct": 0.0,
                "points": [
                    {
                        "budget": budget,
                        "checkpoint_label": label,
                        "AEE": 1.2,
                        "AAE": 6.0,
                        "AAE_Benchmark": 5.8,
                        "DSEC_Fl": 8.0,
                        "total_spikes_g": 80.0,
                    }
                    for budget, label in zip((30, 35, 40), labels, strict=True)
                ],
            }

        aggregation = {
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
        }
        profile_result = {
            "AEE": 1.2,
            "AAE": 6.0,
            "AAE_Benchmark": 5.8,
            "DSEC_Fl": 8.0,
            "total_spikes_g": 80.0,
            "aggregation": aggregation,
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(closure, "RESULTS", Path(temporary)),
            patch.object(closure, "validate_profile", return_value=profile_result),
        ):
            result = closure.convergence_profiles(summary)
            summary["candidates"]["H67"]["rank1_checkpoint_label"] = 40
            with self.assertRaisesRegex(RuntimeError, "derived-field drift"):
                closure.convergence_profiles(summary)
        self.assertEqual(
            result["H67"]["rank1_metrics"]["aggregation"]["pixel_global_mean"][
                "AAE_Benchmark"
            ],
            5.7,
        )
        self.assertEqual(result["NB0"]["rank1_metrics"]["aggregation"]["sequence_count"], 18)

    def test_common_population_requires_exact_list_and_sequence_distribution(self) -> None:
        identity = {
            "validation_file_path": "/fixture/valid.csv",
            "validation_file_sha256": closure.EXPECTED_VALIDATION_LIST_SHA256,
            "frame_count": 825,
            "sequence_frame_counts": dict(closure.EXPECTED_SEQUENCE_FRAME_COUNTS),
        }
        profiles = {
            "Local5": {"population_identity": dict(identity)},
            "H67": {"population_identity": dict(identity)},
            "NB0": {"population_identity": dict(identity)},
        }
        result = closure.validate_common_population(profiles)
        self.assertEqual(result["frame_count"], 825)
        profiles["NB0"]["population_identity"] = {
            **identity,
            "validation_file_sha256": "b" * 64,
        }
        with self.assertRaisesRegex(RuntimeError, "population mismatch"):
            closure.validate_common_population(profiles)

    def test_local_config_identity_binds_active_launch_and_source_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            source = (
                root
                / "h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
            )
            local.mkdir()
            source.parent.mkdir()
            source.write_bytes(b"source")
            state = local / "checkpoint_epoch9_state_dict.pth"
            state.write_bytes(b"state")
            launch_path = local / "active_launch_provenance.json"
            launch_path.write_text(
                json.dumps(
                    {
                        "schema": "local5_active_launch_provenance_v1",
                        "status": "PASS_ACTIVE_CAPTURE",
                        "checks": {"one_active_root_train": True},
                        "artifact_identity": {
                            "source_checkpoint_sha256": sha256(source)
                        },
                    }
                ),
                encoding="utf-8",
            )
            identity_path = local / "training_config_identity.json"
            identity_path.write_text(
                json.dumps(
                    {
                        "schema": "local5_training_config_identity_v1",
                        "status": "PASS",
                        "authority": "ep9_optimizer_scheduler_state",
                        "deterministic_regeneration_equal": True,
                        "config_sha256": "config-sha",
                        "state_sha256": sha256(state),
                        "state_facts": {
                            "state_epoch": 9,
                            "scheduler_last_epoch": 9,
                            "scheduler_milestones": {"13": 1, "20": 1},
                        },
                        "checks": {"runtime": True},
                        "active_launch_provenance": {
                            "path": str(launch_path),
                            "sha256": sha256(launch_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(closure, "LOCAL", local),
                patch.object(closure, "LOCAL_CONFIG_IDENTITY", identity_path),
                patch.object(closure, "LOCAL_ACTIVE_LAUNCH", launch_path),
            ):
                result = closure.validate_local_config_identity()
                self.assertEqual(
                    result["active_launch_provenance_sha256"], sha256(launch_path)
                )
                source.write_bytes(b"replaced")
                with self.assertRaisesRegex(RuntimeError, "launch source checkpoint"):
                    closure.validate_local_config_identity()

    def test_parse_complete_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ranking.md"
            path.write_text(
                "| rank | epoch | AEE |\n|---:|---:|---:|\n"
                "| 1 | 29 | 1.2 |\n| 2 | 24 | 1.3 |\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_ranking(path),
                [{"rank": 1, "epoch": 29}, {"rank": 2, "epoch": 24}],
            )

    def test_rejects_noncontiguous_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ranking.md"
            path.write_text("| 1 | 29 | 1.2 |\n| 3 | 24 | 1.3 |\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid ranking"):
                parse_ranking(path)

    def test_profile_contract_and_population_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint fixture")
            config = root / "config.yml"
            config.write_text("experiment: fixture\n", encoding="utf-8")
            profile = root / "spike_profile.json"
            validation_file = root / "valid.csv"
            validation_file.write_text("fixture\n", encoding="utf-8")
            raw = {
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
                    "AAE_Benchmark": "middlebury_barron_3d_angle_degrees_between_normalized_uv1",
                    "DSEC_Fl": "percentage_epe_gt_3px_and_gt_5pct_of_ground_truth_flow_magnitude",
                    "aggregation": "masked_mean_per_frame_then_equal_mean_over_validation_frames",
                    "population": "local_DSEC_valid_file_list_not_official_hidden_test",
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
                    "sha256": sha256(validation_file),
                },
                "artifact_identity": {
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                    "config_path": str(config.resolve()),
                    "config_sha256": sha256(config),
                },
                "checkpoint_load_audit": {
                    "checkpoint_overlay_keys": 210,
                    "model_overlay_keys": 210,
                    "missing_count": 0,
                    "unexpected_count": 0,
                },
                "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
                "metrics": {
                    "AEE": 1.2,
                    "AAE": 6.0,
                    "AAE_Benchmark": 5.8,
                    "DSEC_Fl": 8.0,
                },
                "total_spikes": 80_000_000_000,
            }
            profile.write_text(json.dumps(raw), encoding="utf-8")
            metrics = validate_profile(
                profile, checkpoint, config, overlay=210, atlif=105, shiftmax=12
            )
            self.assertEqual(metrics["total_spikes_g"], 80.0)
            self.assertEqual(
                metrics["aggregation"]["pixel_global_mean"]["AAE_Benchmark"], 5.7
            )

            raw["metric_contract"]["population"] = "official_hidden_test"
            profile.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "population"):
                validate_profile(profile, checkpoint, config, overlay=210, atlif=105, shiftmax=12)

            raw["metric_contract"]["population"] = (
                "local_DSEC_valid_file_list_not_official_hidden_test"
            )
            raw["samples"] = 824
            profile.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "samples"):
                validate_profile(profile, checkpoint, config, overlay=210, atlif=105, shiftmax=12)

            raw["samples"] = 825
            raw["metric_aggregation_audit"]["sequence_count"] = 17
            profile.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "aggregation_sequences"):
                validate_profile(profile, checkpoint, config, overlay=210, atlif=105, shiftmax=12)

            raw["metric_aggregation_audit"]["sequence_count"] = 18
            del raw["metric_aggregation_audit"]["sequence_balanced_mean"]["AAE"]
            profile.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "aggregation_modes_complete"):
                validate_profile(profile, checkpoint, config, overlay=210, atlif=105, shiftmax=12)

            raw["metric_aggregation_audit"]["sequence_balanced_mean"]["AAE"] = 6.1
            raw["artifact_identity"]["config_sha256"] = "stale-config-sha"
            profile.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "config_sha"):
                validate_profile(profile, checkpoint, config, overlay=210, atlif=105, shiftmax=12)

    def test_local_paired_state_contract_checks_epoch_lr_and_scaler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_lrs = (1.0e-4, 1.0e-4, 5.0e-5, 5.0e-5, 5.0e-6)
            for epoch, factor in ((9, 1.0), (19, 0.5), (29, 0.25)):
                (root / f"checkpoint_epoch{epoch}.pth").write_bytes(
                    f"model-{epoch}".encode()
                )
                lrs = [value * factor for value in base_lrs]
                torch.save(
                    {
                        "epoch": epoch,
                        "optimizer": {"param_groups": [{"lr": lr} for lr in lrs]},
                        "scheduler": {
                            "last_epoch": epoch,
                            "milestones": Counter({13: 1, 20: 1}),
                            "_last_lr": lrs,
                        },
                        "scaler": {"scale": 65536.0},
                    },
                    root / f"checkpoint_epoch{epoch}_state_dict.pth",
                )
            result = validate_local_paired_states(root)
            self.assertEqual(set(result), {"9", "19", "29"})

            state_path = root / "checkpoint_epoch19_state_dict.pth"
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            state["optimizer"]["param_groups"][0]["lr"] = 9.9e-4
            torch.save(state, state_path)
            with self.assertRaisesRegex(RuntimeError, "optimizer LR"):
                validate_local_paired_states(root)

    def test_local_acceptance_binds_rank1_manifest_and_required_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_epoch29.pth"
            checkpoint.write_bytes(b"checkpoint")
            identity = root / "identity.json"
            training_identity = root / "training_config_identity.json"
            training_identity.write_text(
                json.dumps(
                    {
                        "schema": "local5_training_config_identity_v1",
                        "status": "PASS",
                    }
                ),
                encoding="utf-8",
            )
            identity.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": sha256(checkpoint),
                        "best_epoch": 29,
                        "source_bindings": {
                            "training_config_identity": {
                                "path": str(training_identity.resolve()),
                                "sha256": sha256(training_identity),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": sha256(checkpoint),
                        "run_identity_file_sha256": sha256(identity),
                    }
                ),
                encoding="utf-8",
            )
            required = (
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
                "threshold_training_deployment_semantics",
            )
            acceptance = root / "acceptance.json"
            value = {
                "schema": "local5_post_g0_acceptance_v1",
                "accepted": True,
                "samples": 100,
                "blocks": 12,
                "manifest": str(manifest),
                "manifest_sha256": sha256(manifest),
                "run_identity": str(identity),
                "run_identity_sha256": sha256(identity),
                "checks": {name: True for name in required},
            }
            acceptance.write_text(json.dumps(value), encoding="utf-8")
            result = validate_local_acceptance(
                checkpoint, acceptance, training_identity
            )
            self.assertEqual(result["samples"], 100)

            value["checks"]["relation_rtl_binding"] = False
            acceptance.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "required checks"):
                validate_local_acceptance(
                    checkpoint, acceptance, training_identity
                )

    def test_h67_final_binds_profile_trace_audit_and_rtl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_epoch30.pth"
            config = root / "hardware.yml"
            checkpoint.write_bytes(b"checkpoint")
            config.write_bytes(b"config")
            checkpoint_sha = sha256(checkpoint)
            config_sha = sha256(config)

            payload = root / "trace.npz"
            payload.write_bytes(b"trace")
            record = {
                "file": str(payload),
                "sha256": sha256(payload),
                "temporal_tokens": 450,
            }
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "run_context": {
                            "artifact_identity": {
                                "checkpoint_sha256": checkpoint_sha,
                                "config_sha256": config_sha,
                            }
                        },
                        "records": [dict(record) for _ in range(12)],
                    }
                ),
                encoding="utf-8",
            )
            profile = root / "profile.json"
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
            trace_audit = root / "audit.json"
            trace_audit.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "source_manifest": str(manifest),
                        "coverage": {
                            "stages": [0, 1, 2, 3],
                            "four_stage_complete": True,
                        },
                        "records": [{"sha256_ok": True} for _ in range(12)],
                    }
                ),
                encoding="utf-8",
            )
            score = root / "score.json"
            score.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "run_context": {
                            "artifact_identity": {
                                "checkpoint_sha256": checkpoint_sha,
                                "config_sha256": config_sha,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            atlif = root / "atlif.json"
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
            projection = root / "projection.json"
            projection.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "scope": "checkpoint_bound_real_weight_projection_component_rtl_exact_not_full_network",
                        "checkpoint_identity": {
                            "checkpoint_sha256": checkpoint_sha,
                            "config_sha256": config_sha,
                        },
                        "record_count": 12,
                        "required_stage_coverage": [0, 1, 2, 3],
                        "temporal_tokens": 450,
                        "token_id_width": 9,
                        "weight_mode": "checkpoint_dyadic_int8_projection_weight",
                    }
                ),
                encoding="utf-8",
            )
            final = root / "final.json"
            final.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "rank1_epoch": 30,
                        "checkpoint": str(checkpoint),
                        "scope": "checkpoint_bound_component_rtl_exact_not_full_network",
                        "hardware_order_config": str(config),
                        "profile": str(profile),
                        "trace_manifest": str(manifest),
                        "trace_audit": str(trace_audit),
                        "rtl_report": str(score),
                        "atlif_rtl_report": str(atlif),
                        "projection_rtl_report": str(projection),
                    }
                ),
                encoding="utf-8",
            )
            convergence = {"H67": {"rank1_checkpoint_label": 30}}
            # Projection report provenance has a dedicated end-to-end fixture in
            # test_gatestack_projection_report_provenance.py. This closure test
            # isolates the cross-report checkpoint/trace binding contract.
            with (
                patch.object(closure, "H67_FINAL", final),
                patch.object(closure, "validate_projection_provenance") as provenance,
            ):
                result = closure.validate_h67_final(convergence)
                self.assertEqual(result["trace_manifest"], str(manifest))
                provenance.assert_called_once()
                payload.write_bytes(b"stale")
                with self.assertRaisesRegex(RuntimeError, "profile/trace/audit"):
                    closure.validate_h67_final(convergence)


if __name__ == "__main__":
    unittest.main()
