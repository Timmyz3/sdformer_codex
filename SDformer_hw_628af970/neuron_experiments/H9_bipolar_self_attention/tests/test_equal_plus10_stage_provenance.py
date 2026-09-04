from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "entrypoints/run_dsec_fullres_w15_equal_plus10_convergence.py"
)
SPEC = importlib.util.spec_from_file_location("equal_plus10_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EqualPlus10StageProvenanceTest(unittest.TestCase):
    def test_runner_queues_local5_from_its_own_rank1_before_other_candidates(self) -> None:
        self.assertEqual(
            [candidate.name for candidate in MODULE.CANDIDATES],
            ["Local5", "H67", "NB0"],
        )
        self.assertEqual(MODULE.LOCAL5.source_label, 29)
        self.assertEqual(MODULE.LOCAL5.eval_labels, (29, 34, 39))
        self.assertEqual(MODULE.LOCAL5.expected_overlay_keys, 210)
        self.assertEqual(MODULE.LOCAL5.expected_atlif, 105)
        self.assertEqual(MODULE.LOCAL5.expected_shiftmax, 12)
        self.assertEqual(MODULE.LOCAL5.source_model.parent, MODULE.LOCAL5_RUN)

    def test_stage_binds_config_and_rejects_metadata_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yml"
            source_model = root / "source_model.pth"
            source_state = root / "source_state.pth"
            run = root / "run"
            config.write_text("experiment: test\n", encoding="utf-8")
            source_model.write_bytes(b"model fixture")
            torch.save(
                {
                    "epoch": 29,
                    "optimizer": {"param_groups": [{"lr": 2.5e-5}]},
                    "scheduler": {
                        "last_epoch": 29,
                        "milestones": Counter({10: 1, 20: 1}),
                    },
                    "scaler": {"scale": 65536.0},
                },
                source_state,
            )
            candidate = MODULE.Candidate(
                name="fixture",
                config=config,
                source_model=source_model,
                source_state=source_state,
                root=run,
                source_label=30,
                final_label=40,
                eval_labels=(30, 35, 40),
                expected_overlay_keys=0,
                expected_atlif=0,
                expected_shiftmax=0,
            )
            with mock.patch.object(MODULE, "record"):
                MODULE.stage_resume(candidate)
            audit_path = run / "resume_stage_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["config_sha256"], MODULE.sha256(config))
            self.assertEqual(audit["resume_source_budget"], 30)
            self.assertEqual(audit["resume_source_checkpoint_label"], 30)

            audit["config_sha256"] = "drift"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with mock.patch.object(MODULE, "record"):
                with self.assertRaisesRegex(RuntimeError, "metadata drift"):
                    MODULE.stage_resume(candidate)


if __name__ == "__main__":
    unittest.main()
