#!/usr/bin/env python3

from __future__ import annotations

import sys
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from profile_local5_joint_head_trace import (  # noqa: E402
    JOINT_SAMPLING_ID,
    JointHeadOrderedTermTraceSink,
    joint_post_g0_qualification,
    uniform_joint_window,
    validate_joint_plan_freeze_receipt,
)


class Local5JointHeadTraceTest(unittest.TestCase):
    def test_leaf_consumer_validates_plan_freeze_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            runner_path = root / "runner.py"
            receipt_path = root / "receipt.json"
            plan = {"cohort_sha256": "cohort", "records": [{}, {}]}
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            runner_path.write_text("# runner\n", encoding="utf-8")
            runner_sha = hashlib.sha256(runner_path.read_bytes()).hexdigest()
            receipt = {
                "schema": "local5_joint_trace_plan_freeze_receipt_v1",
                "status": "LOCAL_BYTE_ANCHOR_NOT_EXTERNAL_TIMESTAMP",
                "selection_plan": str(plan_path.resolve()),
                "selection_plan_sha256": hashlib.sha256(
                    plan_path.read_bytes()
                ).hexdigest(),
                "selection_plan_git_blob": "0" * 40,
                "generator": str(runner_path.resolve()),
                "generator_sha256": runner_sha,
                "sampling_id": JOINT_SAMPLING_ID,
                "sampling_seed": 20260809,
                "cohort_sha256": "cohort",
                "records": 2,
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            identity = {
                "selection_plan_freeze_receipt": str(receipt_path),
                "selection_plan_freeze_receipt_sha256": hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest(),
            }
            with mock.patch(
                "profile_local5_joint_head_trace.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=plan_path.read_bytes(), stderr=b""
                ),
            ):
                self.assertEqual(
                    validate_joint_plan_freeze_receipt(
                        identity,
                        selection_path=plan_path,
                        selection_sha=receipt["selection_plan_sha256"],
                        plan=plan,
                        runner_binding={"path": str(runner_path), "sha256": runner_sha},
                    ),
                    receipt,
                )
                receipt["selection_plan_sha256"] = "bad"
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                identity["selection_plan_freeze_receipt_sha256"] = hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest()
                with self.assertRaises(ValueError):
                    validate_joint_plan_freeze_receipt(
                        identity,
                        selection_path=plan_path,
                        selection_sha="good",
                        plan=plan,
                        runner_binding={"path": str(runner_path), "sha256": runner_sha},
                    )

    def test_capture_keeps_all_heads_in_one_original_window(self) -> None:
        windows, heads, tokens, lanes = 4, 3, 3, 2
        k = torch.zeros((windows, heads, tokens, 5, lanes), dtype=torch.bool)
        q = torch.zeros((windows, heads, tokens, lanes), dtype=torch.bool)
        gate = torch.zeros((windows, heads, tokens, 5), dtype=torch.long)
        valid = torch.ones((tokens, 5), dtype=torch.bool)
        neighbor = torch.arange(tokens).view(tokens, 1).expand(tokens, 5)
        sink = JointHeadOrderedTermTraceSink(
            groups_per_block_sample=24,
            evidence_level="synthetic",
        )
        sink.capture(
            name="layers.0.swin_blocks.0.attn",
            stage=0,
            block=0,
            sample_id=2,
            k_candidates=k,
            valid=valid,
            gate_code=gate,
            neighbor_index=neighbor,
            q_event=q,
        )
        expected_window = uniform_joint_window(
            batch_windows=windows,
            sample_id=2,
            stage=0,
            block=0,
        )
        self.assertEqual(len(sink.groups), heads)
        self.assertEqual({row["window"] for row in sink.groups}, {expected_window})
        self.assertEqual([row["head"] for row in sink.groups], list(range(heads)))
        self.assertTrue(
            all(
                row["flat_group"] == expected_window * heads + row["head"]
                and row["batch_windows"] == windows
                and row["selection"] == JOINT_SAMPLING_ID
                for row in sink.groups
            )
        )

    def test_formal_qualification_rejects_missing_head(self) -> None:
        stage_heads = (3, 6, 12, 24)
        stage_windows = (440, 120, 30, 10)
        block_depths = (2, 2, 6, 2)
        groups = []
        for stage, depth in enumerate(block_depths):
            for block in range(depth):
                module = f"layers.{stage}.swin_blocks.{block}.attn"
                heads = stage_heads[stage]
                batch_windows = stage_windows[stage]
                for sample in range(100):
                    window = uniform_joint_window(
                        batch_windows=batch_windows,
                        sample_id=sample,
                        stage=stage,
                        block=block,
                    )
                    for head in range(heads):
                        groups.append(
                            {
                                "module": module,
                                "stage": stage,
                                "block": block,
                                "sample": sample,
                                "heads": heads,
                                "batch_windows": batch_windows,
                                "window": window,
                                "head": head,
                                "flat_group": window * heads + head,
                                "tokens": 450,
                                "lanes": 32,
                                "selection": JOINT_SAMPLING_ID,
                            }
                        )
        accepted = joint_post_g0_qualification(
            groups,
            processed_samples=100,
            attached_blocks=12,
            groups_per_block_sample=24,
            run_identity_bound=True,
        )
        self.assertTrue(accepted["qualified"])
        self.assertEqual(accepted["captured_groups"], 13800)
        rejected = joint_post_g0_qualification(
            groups[1:],
            processed_samples=100,
            attached_blocks=12,
            groups_per_block_sample=24,
            run_identity_bound=True,
        )
        self.assertFalse(rejected["qualified"])
        self.assertFalse(rejected["checks"]["exact_same_window_all_heads"])
        wrong_mapping = [dict(group) for group in groups]
        wrong_mapping[0]["heads"] = 6
        rejected_mapping = joint_post_g0_qualification(
            wrong_mapping,
            processed_samples=100,
            attached_blocks=12,
            groups_per_block_sample=24,
            run_identity_bound=True,
        )
        self.assertFalse(rejected_mapping["qualified"])
        self.assertFalse(
            rejected_mapping["checks"]["exact_stage_head_mapping"]
        )


if __name__ == "__main__":
    unittest.main()
