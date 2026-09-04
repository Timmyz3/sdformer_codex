from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from local5_release_receipt import file_sha256, validate_release_receipt


class Local5RankedReleaseReceiptTest(unittest.TestCase):
    def _profile(
        self,
        *,
        config: Path,
        checkpoint: Path,
        deployment: bool,
    ) -> dict:
        stat = checkpoint.stat()
        return {
            "samples": 825,
            "metrics": {"AEE": "1.25"},
            "artifact_identity": {
                "config_path": str(config.resolve()),
                "config_sha256": file_sha256(config),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_size": stat.st_size,
                "checkpoint_mtime_ns": stat.st_mtime_ns,
                "checkpoint_sha256": file_sha256(checkpoint),
            },
            "checkpoint_load_audit": {
                "missing_count": 0,
                "unexpected_count": 0,
                "checkpoint_overlay_keys": 210,
                "model_overlay_keys": 210,
            },
            "eval_protocol": {
                "resolution": [480, 640],
                "crop": None,
                "window_size": [2, 15, 15],
                "remap": "v1",
                "bn_policy": "no_running",
                "eval_batch_size": 1,
            },
            "deployment_contract": (
                {
                    "scope": "attention_core_hardware_order_numeric",
                    "score_quantization": "Q7_step_2^-7",
                    "shiftmax": "Q8_LUT_integer_rowsum_ceil_pow2",
                    "gate_quantization": "Q1.7_RNE",
                    "invalid_candidate_mask": True,
                }
                if deployment
                else None
            ),
        }

    def test_ranked_receipt_binds_selection_and_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint_epoch44.pth"
            training_config = root / "training.yml"
            hardware_config = root / "hardware.yml"
            dyadic_config = root / "dyadic.yml"
            origin_identity = root / "origin_identity.json"
            resume_30_to_40 = root / "resume_30_to_40.json"
            resume_40_to_50 = root / "resume_40_to_50.json"
            origin_state = root / "origin_state.pth"
            source30_model = root / "source30_model.pth"
            source30_state = root / "source30_state.pth"
            source40_model = root / "source40_model.pth"
            source40_state = root / "source40_state.pth"
            ranking = root / "ranking.md"
            convergence = root / "convergence.json"
            float_profile = root / "float.json"
            hardware_profile = root / "hardware.json"
            dyadic_profile = root / "dyadic.json"
            receipt = root / "receipt.json"
            checkpoint.write_bytes(b"checkpoint")
            training_config.write_text("experiment: training\n", encoding="utf-8")
            hardware_config.write_text("experiment: hardware\n", encoding="utf-8")
            dyadic_config.write_text("experiment: dyadic\n", encoding="utf-8")
            for path in (
                origin_state,
                source30_model,
                source30_state,
                source40_model,
                source40_state,
            ):
                path.write_bytes(path.name.encode("ascii"))
            origin_identity.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "state_path": str(origin_state.resolve()),
                        "state_sha256": file_sha256(origin_state),
                        "checks": {"runtime": True},
                    }
                ),
                encoding="utf-8",
            )
            resume_30_to_40.write_text(
                json.dumps(
                    {
                        "scope": (
                            "model_optimizer_scheduler_scaler_resume_not_rng_bit_exact"
                        ),
                        "source_model": str(source30_model.resolve()),
                        "source_model_sha256": file_sha256(source30_model),
                        "source_state": str(source30_state.resolve()),
                        "source_state_sha256": file_sha256(source30_state),
                    }
                ),
                encoding="utf-8",
            )
            resume_40_to_50.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "scope": (
                            "model_optimizer_scheduler_scaler_resume_not_rng_bit_exact"
                        ),
                        "config_sha256": file_sha256(training_config),
                        "config": str(training_config.resolve()),
                        "does_not_inherit_ep29_hardware_provenance": True,
                        "source_model": str(source40_model.resolve()),
                        "source_model_sha256": file_sha256(source40_model),
                        "source_state": str(source40_state.resolve()),
                        "source_state_sha256": file_sha256(source40_state),
                    }
                ),
                encoding="utf-8",
            )
            ranking.write_text(
                "| rank | epoch | AEE |\n|---:|---:|---:|\n| 1 | 44 | 1.2500 |\n",
                encoding="utf-8",
            )
            float_profile.write_text(
                json.dumps(
                    self._profile(
                        config=training_config,
                        checkpoint=checkpoint,
                        deployment=False,
                    )
                ),
                encoding="utf-8",
            )
            hardware_profile.write_text(
                json.dumps(
                    self._profile(
                        config=hardware_config,
                        checkpoint=checkpoint,
                        deployment=True,
                    )
                ),
                encoding="utf-8",
            )
            dyadic = self._profile(
                config=dyadic_config,
                checkpoint=checkpoint,
                deployment=True,
            )
            dyadic["deployment_contract"] = {
                "scope": "attention_core_numeric",
                "score_quantization": "Q7_step_2^-7",
                "shiftmax": "float_exp2",
                "gate_quantization": "Q1.7_RNE",
            }
            dyadic_profile.write_text(json.dumps(dyadic), encoding="utf-8")
            convergence.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "decision": "operationally_plateaued_or_overfit",
                        "rank1_checkpoint_label": 44,
                        "points": [
                            {
                                "checkpoint_label": 44,
                                "AEE": 1.25,
                                "checkpoint_sha256": file_sha256(checkpoint),
                                "profile": str(float_profile.resolve()),
                                "profile_sha256": file_sha256(float_profile),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            value = {
                "schema": "local5_ranked_checkpoint_release_receipt_v1",
                "status": "PASS",
                "watcher_session_uuid": "test-session",
                "selection_metric": "AEE",
                "selection_decision": "operationally_plateaued_or_overfit",
                "best_epoch": 44,
            }
            for stem, path in {
                "ranking": ranking,
                "convergence_summary": convergence,
                "checkpoint": checkpoint,
                "training_config": training_config,
                "origin_training_identity": origin_identity,
                "resume_30_to_40": resume_30_to_40,
                "resume_40_to_50": resume_40_to_50,
                "dyadic_config": dyadic_config,
                "dyadic_profile": dyadic_profile,
                "config": hardware_config,
                "float_profile": float_profile,
                "hardware_profile": hardware_profile,
            }.items():
                value[f"{stem}_path"] = str(path.resolve())
                value[f"{stem}_sha256"] = file_sha256(path)
            receipt.write_text(json.dumps(value), encoding="utf-8")

            self.assertEqual(validate_release_receipt(receipt), value)

            value["best_epoch"] = 49
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection contract"):
                validate_release_receipt(receipt)

    def test_legacy_receipt_remains_accepted(self) -> None:
        receipt = (
            ROOT
            / "results/local5_fullres_bb1e4_postg0_profile100_20260805/"
            "post_g0_release_receipt.json"
        )
        if not receipt.is_file():
            self.skipTest("legacy evidence package is not present")
        self.assertEqual(
            validate_release_receipt(receipt).get("schema"),
            "local5_release_receipt_v2",
        )


if __name__ == "__main__":
    unittest.main()
