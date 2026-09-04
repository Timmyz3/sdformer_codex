from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_local5_joint_head_profile as runner


class RunLocal5JointHeadProfileTest(unittest.TestCase):
    def test_stop_process_group_reaps_zombie_leader(self) -> None:
        process = subprocess.Popen(["/bin/sh", "-c", "exit 0"], start_new_session=True)
        state_path = Path(f"/proc/{process.pid}/stat")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if state_path.is_file() and state_path.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            self.fail("测试子进程没有进入zombie状态")

        runner.stop_process_group(process, timeout_seconds=0.5)

        self.assertIsNotNone(process.returncode)
        self.assertFalse(state_path.exists())

    def test_pid_namespace_mapping_only_allows_outermost_host_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / "123").mkdir()
            (proc / "123" / "status").write_text(
                "Name:\tpython\nNSpid:\t619733\t123\n",
                encoding="utf-8",
            )
            (proc / "456").mkdir()
            (proc / "456" / "status").write_text(
                "Name:\tworker\n",
                encoding="utf-8",
            )

            host_pids = runner.pid_namespace_host_pids({123, 456}, proc_root=proc)

            self.assertEqual(host_pids, {456, 619733})
            self.assertNotIn(123, host_pids)

    def test_unmapped_gpu_pid_claim_is_single_and_sticky(self) -> None:
        claimed, foreign = runner.classify_gpu_processes(
            {687421}, {3770126, 3770158}, None, child_alive=True
        )
        self.assertEqual(claimed, 687421)
        self.assertEqual(foreign, set())

        same_claim, foreign = runner.classify_gpu_processes(
            {687421, 999999}, {3770126, 3770158}, claimed, child_alive=True
        )
        self.assertEqual(same_claim, claimed)
        self.assertEqual(foreign, {999999})

        same_claim, foreign = runner.classify_gpu_processes(
            {999999}, {3770126, 3770158}, claimed, child_alive=True
        )
        self.assertEqual(same_claim, claimed)
        self.assertEqual(foreign, {999999})

        same_claim, foreign = runner.classify_gpu_processes(
            {687421, 3770158}, {3770126, 3770158}, claimed, child_alive=True
        )
        self.assertEqual(same_claim, claimed)
        self.assertEqual(foreign, {687421})

    def test_unmapped_claim_rejects_ambiguous_or_direct_mapped_foreign(self) -> None:
        claimed, foreign = runner.classify_gpu_processes(
            {1, 2}, {100}, None, child_alive=True
        )
        self.assertIsNone(claimed)
        self.assertEqual(foreign, {1, 2})

        claimed, foreign = runner.classify_gpu_processes(
            {100, 2}, {100}, None, child_alive=True
        )
        self.assertIsNone(claimed)
        self.assertEqual(foreign, {2})

    def test_selection_plan_is_write_once_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort.json"
            plan = root / "plan.json"
            cohort.write_text(json.dumps({"cohort_sha256": "abc"}), encoding="utf-8")
            with (
                mock.patch.object(runner, "OUTPUT", root),
                mock.patch.object(runner, "SELECTION_PLAN", plan),
                mock.patch.object(runner, "COHORT_MANIFEST", cohort),
            ):
                expected = runner.write_selection_plan()
                original = plan.read_bytes()
                self.assertEqual(json.loads(original), expected)
                self.assertEqual(runner.write_selection_plan(), expected)
                self.assertEqual(plan.read_bytes(), original)

                plan.write_bytes(original + b" ")
                with self.assertRaises(RuntimeError):
                    runner.write_selection_plan()
                self.assertEqual(plan.read_bytes(), original + b" ")

    def test_plan_receipt_binds_plan_and_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            receipt_path = root / "receipt.json"
            plan = {
                "cohort_sha256": "cohort",
                "records": [{} for _ in range(3)],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            receipt = {
                "schema": "local5_joint_trace_plan_freeze_receipt_v1",
                "status": "LOCAL_BYTE_ANCHOR_NOT_EXTERNAL_TIMESTAMP",
                "selection_plan": str(plan_path.resolve()),
                "selection_plan_sha256": runner.sha256(plan_path),
                "selection_plan_git_blob": "0" * 40,
                "generator": str(Path(runner.__file__).resolve()),
                "generator_sha256": runner.sha256(Path(runner.__file__).resolve()),
                "sampling_id": runner.SAMPLING_ID,
                "sampling_seed": runner.SAMPLING_SEED,
                "cohort_sha256": "cohort",
                "records": 3,
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with (
                mock.patch.object(runner, "SELECTION_PLAN", plan_path),
                mock.patch.object(runner, "PLAN_FREEZE_RECEIPT", receipt_path),
                mock.patch.object(
                    runner.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        args=[], returncode=0, stdout=plan_path.read_bytes(), stderr=b""
                    ),
                ),
            ):
                self.assertEqual(runner.validate_plan_freeze_receipt(plan), receipt)
                receipt["selection_plan_sha256"] = "bad"
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    runner.validate_plan_freeze_receipt(plan)


if __name__ == "__main__":
    unittest.main()
