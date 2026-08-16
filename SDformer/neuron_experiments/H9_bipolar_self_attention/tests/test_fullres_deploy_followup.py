from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

from run_dsec_fullres_paper_w15_deploy_followup import reusable_profile


class FullresDeployFollowupTest(unittest.TestCase):
    def test_reuses_only_matching_audited_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yml"
            checkpoint = root / "checkpoint.pth"
            output = root / "output"
            output.mkdir()
            contract = {
                "scope": "attention_core_hardware_order_numeric",
                "full_network_fixed_point": False,
            }
            config.write_text(
                yaml.safe_dump({"runtime": {"deployment_contract": contract}}),
                encoding="utf-8",
            )
            checkpoint.write_bytes(b"checkpoint")
            stat = checkpoint.stat()
            profile = {
                "metrics": {
                    "AEE": 1.0,
                    "AAE": 2.0,
                    "AAE_Benchmark": 1.5,
                    "AEE_PE1": 0.1,
                    "AEE_PE2": 0.2,
                    "AEE_outliers": 0.3,
                },
                "total_spikes": 1_000_000_000,
                "global_firing_rate": 0.1,
                "energy_uj": 1.0,
                "samples": 825,
                "eval_protocol": {
                    "resolution": [480, 640],
                    "crop": None,
                    "window_size": [2, 15, 15],
                    "remap": "v1",
                    "bn_policy": "no_running",
                    "eval_batch_size": 1,
                },
                "checkpoint_load_audit": {
                    "missing_count": 0,
                    "unexpected_count": 0,
                },
                "artifact_identity": {
                    "config_path": str(config.resolve()),
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_size": stat.st_size,
                    "checkpoint_mtime_ns": stat.st_mtime_ns,
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                },
                "deployment_contract": contract,
            }
            profile_path = output / "spike_profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            reused = reusable_profile(config, checkpoint, output)
            self.assertIsNotNone(reused)
            self.assertEqual(reused[1]["samples"], 825)

            checkpoint.write_bytes(b"changed checkpoint")
            self.assertIsNone(reusable_profile(config, checkpoint, output))


if __name__ == "__main__":
    unittest.main()
