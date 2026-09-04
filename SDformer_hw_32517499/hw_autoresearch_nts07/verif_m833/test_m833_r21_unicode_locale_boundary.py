#!/usr/bin/env python3
"""Source-only Unicode boundary regression for C2 R21. No EDA/tool probes."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent
PY36 = Path("/usr/libexec/platform-python3.6")
GUARD = HW / "verif_m826/m826_c2_r20_atomic_guard.py"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m833_c2_r21_unicode_exact_sha.sh"
CONTRACT = HW / "contracts/m833_c2_r21_unicode_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m833_c2_r21_unicode_vcs_launch_candidate_source_only_r1_20260829.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def clean(extra=None):
    value = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    if extra:
        value.update(extra)
    return value


def validate(env):
    return subprocess.run([
        str(PY36), str(GUARD), "validate-source", "--hw-root", str(HW),
        "--contract", str(CONTRACT), "--candidate", str(CANDIDATE),
        "--runner", str(RUNNER),
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
       universal_newlines=True, timeout=120)


class M833UnicodeLocaleBoundary(unittest.TestCase):
    def test_real_absolute_chinese_path_is_frozen_and_present(self):
        self.assertTrue(DOCS359.is_absolute())
        self.assertTrue(DOCS359.is_file())
        self.assertTrue(any(ord(ch) > 127 for ch in str(DOCS359)))
        self.assertEqual(hashlib.sha256(DOCS359.read_bytes()).hexdigest(),
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")

    def test_outer_clean_c_unwrapped_python36_reproduces_failure(self):
        process = validate(clean())
        self.assertEqual(process.returncode, 1)
        self.assertIn("UnicodeEncodeError", process.stderr)
        self.assertIn("ascii", process.stderr)

    def test_pythonutf8_is_explicitly_not_a_fix_on_this_python36(self):
        encoding = subprocess.run([
            str(PY36), "-c", "import sys; print(sys.getfilesystemencoding())"
        ], env=clean({"PYTHONUTF8": "1"}), stdout=subprocess.PIPE,
           stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
        self.assertEqual(encoding.returncode, 0)
        self.assertEqual(encoding.stdout.strip(), "ascii")
        process = validate(clean({"PYTHONUTF8": "1"}))
        self.assertEqual(process.returncode, 1)
        self.assertIn("UnicodeEncodeError", process.stderr)

    def test_runner_local_c_utf8_passes_under_outer_clean_c(self):
        wrapped = clean({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
        encoding = subprocess.run([
            str(PY36), "-c", "import sys; print(sys.getfilesystemencoding())"
        ], env=wrapped, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
           universal_newlines=True, timeout=30)
        self.assertEqual(encoding.returncode, 0)
        self.assertEqual(encoding.stdout.strip(), "utf-8")
        process = validate(wrapped)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("PASS_M826_R20_SOURCE_IDENTITY__NO_VCS_OR_EDA",
                      process.stdout)

    def test_runner_wraps_every_python_execution_but_not_vcs_or_simv(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'env LANG=C.UTF-8 LC_ALL=C.UTF-8 "${python36}" "$@"', text)
        self.assertNotIn("PYTHONUTF8", text)
        self.assertNotIn("PYTHONIOENCODING", text)
        direct = [line.strip() for line in text.splitlines()
                  if '"${python36}"' in line]
        self.assertEqual(direct, [
            'env LANG=C.UTF-8 LC_ALL=C.UTF-8 "${python36}" "$@"',
            'expect_file_sha "${python36}" 9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f',
        ])
        self.assertEqual(text.count('python36_utf8 "${guard}"'), 12)
        self.assertEqual(text.count("python36_utf8 - \"${work}\""), 1)
        self.assertNotIn("export LANG", text)
        self.assertNotIn("export LC_ALL", text)
        self.assertIn('"${vcs}" -full64', text)
        self.assertIn('"${phase_dir}/simv" "+ntb_random_seed=${seed}"', text)


if __name__ == "__main__":
    unittest.main()
