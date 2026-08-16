from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evidence_provenance import (
    sha256_file,
    validate_motion_rqtb_provenance,
    validate_motion_tesc_provenance,
)


def binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


class MotionModelProvenanceTest(unittest.TestCase):
    def test_tesc_and_rqtb_fail_closed_on_profile_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {}
            for name in (
                "profile",
                "config",
                "checkpoint",
                "analyzer",
                "model",
                "validator",
                "watcher",
                "test_log",
                "test_a",
                "test_b",
                "test_c",
            ):
                path = root / f"{name}.txt"
                path.write_text(name + "\n", encoding="utf-8")
                files[name] = path

            source = {
                "profile": str(files["profile"].resolve()),
                "config_path": str(files["config"].resolve()),
                "config_sha256": sha256_file(files["config"]),
                "checkpoint_path": str(files["checkpoint"].resolve()),
                "checkpoint_sha256": sha256_file(files["checkpoint"]),
                "temporal_tokens": 450,
            }
            test_bindings = [
                binding(files["test_a"]),
                binding(files["test_b"]),
                binding(files["test_c"]),
            ]
            tesc = {
                "schema": "motion_temporal_equivalence_v2",
                "profile": str(files["profile"].resolve()),
                "source": source,
                "provenance": {
                    "profile": binding(files["profile"]),
                    "config": binding(files["config"]),
                    "checkpoint": binding(files["checkpoint"]),
                    "analyzer": binding(files["analyzer"]),
                    "validator": binding(files["validator"]),
                    "watcher": binding(files["watcher"]),
                    "test_log": binding(files["test_log"]),
                    "tests": test_bindings,
                },
            }
            validate_motion_tesc_provenance(tesc)
            tesc_path = root / "tesc.json"
            tesc_path.write_text(json.dumps(tesc), encoding="utf-8")

            rqtb = {
                "schema": "motion_reversible_quotient_bundle_v2",
                "source": source,
                "provenance": {
                    "profile": binding(files["profile"]),
                    "tesc_report": binding(tesc_path),
                    "config": binding(files["config"]),
                    "checkpoint": binding(files["checkpoint"]),
                    "model": binding(files["model"]),
                    "validator": binding(files["validator"]),
                    "watcher": binding(files["watcher"]),
                    "test_log": binding(files["test_log"]),
                    "tests": test_bindings,
                },
            }
            validate_motion_rqtb_provenance(rqtb)

            files["profile"].write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA drift"):
                validate_motion_tesc_provenance(tesc)
            with self.assertRaisesRegex(RuntimeError, "SHA drift"):
                validate_motion_rqtb_provenance(rqtb)


if __name__ == "__main__":
    unittest.main()
