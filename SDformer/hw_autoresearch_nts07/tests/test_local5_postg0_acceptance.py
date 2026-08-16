from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_local5_postg0_acceptance as acceptance


class Local5PostG0AcceptanceTest(unittest.TestCase):
    def _run_with_reports(
        self,
        *,
        replay_report: dict[str, object],
        descriptor_report: dict[str, object],
    ) -> dict[str, object]:
        manifest = {
            "qualification": {"qualified": True},
            "groups": [{"sample": 0}],
            "run_identity_file_sha256": "identity-hash",
            "payload_file": "payload.npz",
            "payload_sha256": "payload-hash",
            "cohort_file": "cohort.json",
            "cohort_file_sha256": "cohort-hash",
            "checkpoint": "/checkpoint.pth",
            "checkpoint_sha256": "checkpoint-hash",
            "projection_contract_file": "projection.json",
            "projection_contract_file_sha256": "projection-manifest-hash",
            "projection_contract_payload": "projection.npz",
            "projection_contract_payload_sha256": "projection-payload-hash",
            "threshold_training_semantics": {
                "threshold_modes": ["official_atlif"],
                "homeostatic_freeze_after_step": 1224,
                "homeostatic_update_frozen_after_boundary": True,
                "optimizer_gradient_freeze_enabled": False,
                "optimizer_threshold_lr": 5.0e-6,
                "official_atlif_runtime_clamp_applied": False,
                "inference_threshold_source": "checkpoint_static_parameter",
            },
        }
        analysis = {
            "manifest_sha256": "manifest-hash",
            "formal_coverage": {"samples": 100, "blocks": 12},
            "groups": 1,
            "descriptors": 450,
        }
        replay_recomputed = {
            "groups": 1,
            "manifest": "",
            "manifest_sha256": "manifest-hash",
            "run_identity_file_sha256": "identity-hash",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "manifest.json"
            replay_path = root / "replay.json"
            descriptor_path = root / "descriptor.json"
            identity_path = root / "identity.json"
            projection_manifest_path = root / "projection.json"
            projection_payload_path = root / "projection.npz"
            replay_report = dict(replay_report)
            if replay_report.get("manifest") == "__CURRENT__":
                replay_report["manifest"] = str(manifest_path.resolve())
            manifest_path.write_text("{}", encoding="utf-8")
            identity_path.write_text(
                json.dumps(
                    {
                    "schema": "local5_post_g0_run_identity_v3",
                        "release_receipt": str(root / "receipt.json"),
                        "release_receipt_sha256": "receipt-hash",
                    }
                ),
                encoding="utf-8",
            )
            projection_payload_path.write_bytes(b"projection-payload")
            projection_manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "local5_checkpoint_projection_contract_v2",
                        "status": "THETA_FOLDED_WEIGHT_CONTRACT",
                        "topology_contract": acceptance.TOPOLOGY_CONTRACT,
                        "checkpoint": "/checkpoint.pth",
                        "checkpoint_sha256": "checkpoint-hash",
                        "payload_file": projection_payload_path.name,
                        "payload_sha256": "projection-payload-hash",
                        "blocks": [{"theta": 1.0} for _ in range(12)],
                        "value_contract": (
                            "V=K_binary_event*theta_K(block); theta_K is folded into "
                            "projection W before dyadic INT8 quantization"
                        ),
                        "quantization_order": (
                            "W_eff=theta_K*W_float; quantize_dyadic_int8(W_eff)"
                        ),
                        "runtime_datapath": (
                            "K remains a 1-bit event; no runtime theta multiplier or "
                            "event-width increase"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            replay_recomputed["manifest"] = str(manifest_path.resolve())
            replay_path.write_text(
                json.dumps(replay_report),
                encoding="utf-8",
            )
            descriptor_path.write_text(
                json.dumps(descriptor_report),
                encoding="utf-8",
            )
            with (
                patch.object(
                    acceptance,
                    "load_trace",
                    return_value=(manifest, {}),
                ),
                patch.object(
                    acceptance,
                    "analyze",
                    return_value=analysis,
                ),
                patch.object(
                    acceptance,
                    "replay_trace",
                    return_value={
                        "groups": 1,
                    },
                ),
                patch.object(
                    acceptance,
                    "file_sha256",
                    side_effect=lambda path: (
                        "manifest-hash"
                        if Path(path) == manifest_path
                        else "identity-hash"
                        if Path(path) == identity_path
                        else "projection-manifest-hash"
                        if Path(path) == projection_manifest_path
                        else "projection-payload-hash"
                        if Path(path) == projection_payload_path
                        else "report-hash"
                    ),
                ),
                patch.object(
                    acceptance,
                    "verify_contract",
                    return_value={
                        "status": "PASS",
                        "checkpoint_sha256": "checkpoint-hash",
                        "blocks": 12,
                    },
                ),
            ):
                return acceptance.validate(
                    manifest_path=manifest_path,
                    replay_report_path=replay_path,
                    descriptor_report_path=descriptor_path,
                    run_identity_path=identity_path,
                )

    def test_accepts_bound_threshold_semantics(self) -> None:
        analysis = {
            "manifest_sha256": "manifest-hash",
            "formal_coverage": {"samples": 100, "blocks": 12},
            "groups": 1,
            "descriptors": 450,
        }
        replay = {
            "groups": 1,
            "manifest": "__CURRENT__",
            "manifest_sha256": "manifest-hash",
            "run_identity_file_sha256": "identity-hash",
        }
        result = self._run_with_reports(
            replay_report=replay,
            descriptor_report=analysis,
        )
        self.assertTrue(result["accepted"])
        self.assertTrue(
            result["checks"]["threshold_training_deployment_semantics"]
        )
        self.assertTrue(
            result["checks"]["checkpoint_projection_payload_recomputed"]
        )

    def test_rejects_tampered_descriptor_report(self) -> None:
        replay = {
            "groups": 1,
            "manifest": "/unused",
            "manifest_sha256": "manifest-hash",
            "run_identity_file_sha256": "identity-hash",
        }
        with self.assertRaisesRegex(ValueError, "descriptor"):
            self._run_with_reports(
                replay_report=replay,
                descriptor_report={"tampered": True},
            )

    def test_rejects_tampered_replay_report(self) -> None:
        analysis = {
            "manifest_sha256": "manifest-hash",
            "formal_coverage": {"samples": 100, "blocks": 12},
            "groups": 1,
            "descriptors": 450,
        }
        with self.assertRaisesRegex(ValueError, "replay"):
            self._run_with_reports(
                replay_report={"groups": 999},
                descriptor_report=analysis,
            )

    def test_rejects_threshold_gradient_freeze_semantic_drift(self) -> None:
        manifest = {
            "threshold_training_semantics": {
                "threshold_modes": ["official_atlif"],
                "homeostatic_freeze_after_step": 1224,
                "homeostatic_update_frozen_after_boundary": True,
                "optimizer_gradient_freeze_enabled": True,
                "optimizer_threshold_lr": 5.0e-6,
                "official_atlif_runtime_clamp_applied": False,
                "inference_threshold_source": "checkpoint_static_parameter",
            }
        }
        with self.assertRaisesRegex(ValueError, "语义漂移"):
            acceptance.validate_threshold_semantics(manifest)


if __name__ == "__main__":
    unittest.main()
