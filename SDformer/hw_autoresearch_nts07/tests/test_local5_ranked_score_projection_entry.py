from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = (
    ROOT / "sim_new_arch/run_local5_score_projection_checks_ranked.sh",
    ROOT / "sim_new_arch/run_local5_qgasr2c_fivebank_checks_ranked.sh",
)


class Local5RankedScoreProjectionEntryTest(unittest.TestCase):
    def test_entry_is_non_destructive(self) -> None:
        for entry in ENTRIES:
            with self.subTest(entry=entry.name):
                self.assertNotIn("rm -rf", entry.read_text(encoding="utf-8"))

    def test_existing_build_directory_fails_closed(self) -> None:
        for entry in ENTRIES:
            with self.subTest(entry=entry.name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                build = root / "existing-build"
                build.mkdir()
                env = os.environ.copy()
                env.update(
                    {
                        "BUILD_DIR": str(build),
                        "BUILD_ROOT": str(build),
                        "RESULT_DIR": str(root / "result"),
                        "VECTOR_DIR": str(root / "vectors"),
                        "POSTSCORE_REPORT": str(root / "report.json"),
                    }
                )
                completed = subprocess.run(
                    ["bash", str(entry)],
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("build directory already exists", completed.stderr)


if __name__ == "__main__":
    unittest.main()
