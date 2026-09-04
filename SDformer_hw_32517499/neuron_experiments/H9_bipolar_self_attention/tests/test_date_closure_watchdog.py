from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

import supervise_date_closure_watchers_20260805 as watchdog  # noqa: E402


class DateClosureWatchdogTest(unittest.TestCase):
    def test_all_required_followers_have_distinct_completion_contracts(self) -> None:
        names = [task.name for task in watchdog.TASKS]
        self.assertEqual(len(names), 6)
        self.assertEqual(len(set(names)), 6)
        self.assertEqual(
            set(names),
            {
                "local5_ep9_config_identity",
                "local5_checkpoint_bound_rtl",
                "h67_nb0_equal_plus10",
                "h67_ep30_component_rtl",
                "h67_postconvergence_component_rtl",
                "date_algorithm_closure",
            },
        )
        self.assertTrue(all(task.pid_file != watchdog.PID_FILE for task in watchdog.TASKS))

    def test_json_and_marker_completion_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "identity.json"
            marker_path = root / "status.log"
            json_task = watchdog.Task(
                "json", ("python", "task.py"), root / "json.pid", root / "json.log",
                json_path, completion_json_status="PASS",
            )
            marker_task = watchdog.Task(
                "marker", ("python", "task.py"), root / "marker.pid", root / "marker.log",
                marker_path, completion_marker="ALL COMPLETE",
                required_paths=(root / "artifact.json",),
            )
            self.assertFalse(watchdog.task_complete(json_task))
            json_path.write_text(json.dumps({"status": "PENDING"}), encoding="utf-8")
            self.assertFalse(watchdog.task_complete(json_task))
            json_path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            self.assertTrue(watchdog.task_complete(json_task))
            marker_path.write_text("WAIT\n", encoding="utf-8")
            self.assertFalse(watchdog.task_complete(marker_task))
            marker_path.write_text("WAIT\nALL COMPLETE\n", encoding="utf-8")
            self.assertFalse(watchdog.task_complete(marker_task))
            (root / "artifact.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue(watchdog.task_complete(marker_task))

    def test_pid_must_match_expected_script(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = watchdog.Task(
                "task", ("python", "-u", "/tmp/expected.py"), root / "task.pid",
                root / "task.log", root / "done.log", completion_marker="DONE",
            )
            task.pid_file.write_text("123\n", encoding="utf-8")
            with (
                patch.object(watchdog, "pid_cmdline", return_value=("python", "/tmp/other.py")),
                patch.object(watchdog, "matching_pids", return_value=[]),
            ):
                self.assertFalse(watchdog.task_alive(task))
            with patch.object(watchdog, "pid_cmdline", return_value=("python", "/tmp/expected.py")):
                self.assertTrue(watchdog.task_alive(task))

    def test_stale_pid_file_adopts_matching_detached_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = watchdog.Task(
                "task", ("python", "-u", "/tmp/expected.py"), root / "task.pid",
                root / "task.log", root / "done.log", completion_marker="DONE",
            )
            task.pid_file.write_text("123\n", encoding="utf-8")
            with (
                patch.object(watchdog, "pid_cmdline", return_value=None),
                patch.object(watchdog, "matching_pids", return_value=[456]),
            ):
                self.assertTrue(watchdog.task_alive(task))
            self.assertEqual(task.pid_file.read_text(encoding="utf-8"), "456\n")

    def test_reap_children_is_nonblocking_and_exhaustive(self) -> None:
        with patch.object(
            watchdog.os,
            "waitpid",
            side_effect=[(123, 0), (456, 0), (0, 0)],
        ) as waitpid:
            self.assertEqual(watchdog.reap_children(), 2)
        self.assertEqual(waitpid.call_count, 3)
        waitpid.assert_called_with(-1, watchdog.os.WNOHANG)

    def test_reap_children_accepts_no_children(self) -> None:
        with patch.object(watchdog.os, "waitpid", side_effect=ChildProcessError):
            self.assertEqual(watchdog.reap_children(), 0)


if __name__ == "__main__":
    unittest.main()
