#!/usr/bin/env python3
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent
GUARD = HERE / "m859_c2_r25_shared_whitelist_guard.py"
CONTRACT = HW / "contracts/m859_c2_r25_shared_whitelist_source_only_contract_r1_20260829.json"
CANDIDATE = HW / "contracts/m859_c2_r25_shared_whitelist_vcs_launch_candidate_source_only_r1_20260829.json"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m859_c2_r25_shared_whitelist_exact_sha.sh"


class SourceClosureTest(unittest.TestCase):
    def test_exact_source_identity(self):
        subprocess.run([
            "/usr/libexec/platform-python3.6", str(GUARD),
            "validate-source", "--hw-root", str(HW),
            "--contract", str(CONTRACT), "--candidate", str(CANDIDATE),
            "--runner", str(RUNNER),
        ], check=True, env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
           stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()
