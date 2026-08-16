from __future__ import annotations

import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_local5_ep44_12block_job_replay import parse_log, validate_plan


class Local5Ep44TwelveBlockReportTest(unittest.TestCase):
    def test_parse_rejects_failure_before_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulation.log"
            blocks = "".join(
                f"BLOCK ordinal={index} stage=0 block=0 group={index} "
                "empty=0 cycles=100 results=900\n"
                for index in range(12)
            )
            path.write_text(
                "ERROR: injected\n"
                + blocks
                + "PASS Local5 ep44 12-block tagged jobs seed=23133 "
                "cycles=1200 jobs=12 token=5400 weight=768 result=10800 "
                "result_stall=1 token_stall=1 weight_stall=1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "failure marker"):
                parse_log(path)

    def test_runner_is_non_destructive(self) -> None:
        runner = ROOT / "sim_qfit/run_local5_ep44_12block_job_replay.sh"
        self.assertNotIn("rm -rf", runner.read_text(encoding="utf-8"))

    def test_plan_rejects_drifted_source_vector_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vector_dir = root / "vectors"
            plan_dir = root / "plan"
            vector_dir.mkdir()
            plan_dir.mkdir()
            payload = vector_dir / "input_q.memh"
            payload.write_text("00\n", encoding="ascii")
            payload_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
            vector_manifest = vector_dir / "manifest.json"
            vector_manifest.write_text(
                json.dumps(
                    {
                        "artifacts": {
                            "input_q": {
                                "file": payload.name,
                                "sha256": payload_sha,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            artifact = plan_dir / "selected_group.memh"
            artifact.write_text("00\n", encoding="ascii")
            artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            plan = plan_dir / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema": "local5_ep44_12block_job_plan_v1",
                        "status": "PASS",
                        "jobs": 12,
                        "nonempty_jobs": 10,
                        "rows": [{} for _ in range(12)],
                        "artifacts": {
                            "group": {
                                "file": artifact.name,
                                "sha256": artifact_sha,
                            }
                        },
                        "source_vector_manifest": str(vector_manifest),
                        "source_vector_manifest_sha256": hashlib.sha256(
                            vector_manifest.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            payload.write_text("ff\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "source vector artifact drift"):
                validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
