from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

from run_h60_family_deploy_eval import profile_artifact_status as deploy_status
from run_h9_standard_valid825_eval import profile_artifact_status as standard_status


class ProfileArtifactIdentityTest(unittest.TestCase):
    def test_match_legacy_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yml"
            checkpoint = root / "checkpoint.pth"
            profile = root / "spike_profile.json"
            config.write_text("test: true\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            stat = checkpoint.stat()
            identity = {
                "config_path": str(config.resolve()),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_size": stat.st_size,
                "checkpoint_mtime_ns": stat.st_mtime_ns,
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
            profile.write_text(
                json.dumps({"artifact_identity": identity}), encoding="utf-8"
            )
            self.assertEqual(standard_status(profile, config, checkpoint), "match")
            self.assertEqual(deploy_status(profile, config, checkpoint), "match")

            profile.write_text("{}", encoding="utf-8")
            self.assertEqual(standard_status(profile, config, checkpoint), "legacy")
            self.assertEqual(deploy_status(profile, config, checkpoint), "legacy")

            profile.write_text(
                json.dumps({"artifact_identity": identity}), encoding="utf-8"
            )
            config.write_text("test: false\n", encoding="utf-8")
            self.assertEqual(standard_status(profile, config, checkpoint), "mismatch")
            self.assertEqual(deploy_status(profile, config, checkpoint), "mismatch")


if __name__ == "__main__":
    unittest.main()
