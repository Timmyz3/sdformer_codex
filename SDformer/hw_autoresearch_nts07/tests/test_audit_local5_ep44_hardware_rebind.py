from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_local5_ep44_hardware_rebind as audit


class Local5Ep44FinalAuditTest(unittest.TestCase):
    def test_sealed_files_fail_on_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.log"
            artifact.write_text("sealed\n", encoding="utf-8")
            complete = {"files": {artifact.name: audit.file_sha256(artifact)}}
            with patch.object(audit, "INTEGRATED_DIR", root):
                self.assertTrue(audit.sealed_files_match(complete))
                artifact.write_text("drifted\n", encoding="utf-8")
                self.assertFalse(audit.sealed_files_match(complete))

    def test_vector_artifacts_fail_on_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            artifact = root / "input.memh"
            artifact.write_text("00\n", encoding="utf-8")
            manifest = {
                "artifacts": {
                    "input": {
                        "file": artifact.name,
                        "sha256": audit.file_sha256(artifact),
                    }
                }
            }
            with patch.object(audit, "VECTOR_MANIFEST", manifest_path):
                self.assertTrue(audit.vector_artifacts_match(manifest))
                artifact.write_text("01\n", encoding="utf-8")
                self.assertFalse(audit.vector_artifacts_match(manifest))


if __name__ == "__main__":
    unittest.main()
