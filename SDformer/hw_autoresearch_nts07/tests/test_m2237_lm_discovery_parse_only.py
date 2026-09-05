"""Critical recovery mutations operate in memory on immutable original logs."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

HW = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m2237", HW / "system_simulator/scripts/check_m2237_m2223_lm_discovery_parse_only.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
READ = Path.read_text


class RecoveryTest(unittest.TestCase):
    def mutated(self, relative, change):
        target = m.RAW_DIRECTORY / relative
        def reader(path, *args, **kwargs):
            value = READ(path, *args, **kwargs)
            return change(value) if path == target else value
        with patch.object(Path, "read_text", reader), self.assertRaises(m.Failure):
            m.validate(m.RAW_DIRECTORY)

    def test_raw_echo_and_authenticated_rename_accept(self):
        result = m.validate(m.RAW_DIRECTORY)
        self.assertTrue(result["milkyway_exec_set_readback"]["exact"])
        self.assertFalse(result["claim_boundary"]["library_conversion"])
        self.assertEqual(result["authenticated_relocation"]["recorded_staging"], str(m.STAGING_DIRECTORY))

    def test_anchored_runtime_fatal_rejected(self):
        for suffix in ["\nM2221_FATAL_FAIL_CLOSED: injected\n", "\n  M2221_FATAL_FAIL_CLOSED: injected\n"]:
            self.mutated("lm_discovery.log", lambda text: text + suffix)

    def test_duplicate_and_missing_runtime_markers(self):
        for prefix in ["M2221_STARTUP ", "M2221_COMMAND ", "M2221_OPTION ",
                       "M2221_MILKYWAY_SET ", "M2221_NO_SIDE_EFFECTS ", m.RAW_PASS]:
            self.mutated("lm_discovery.log", lambda text: text + "\n" + next(
                line for line in text.splitlines() if line.startswith(prefix)) + "\n")
        self.mutated("lm_discovery.log", lambda text: text.replace("\n" + m.RAW_PASS + "\n", "\n"))

    def test_unauthenticated_mapping_rejected(self):
        old = str(m.STAGING_DIRECTORY / "isolated_cwd").encode().hex()
        new = str(m.RAW_DIRECTORY / "isolated_cwd").encode().hex()
        self.mutated("lm_discovery.log", lambda text: text.replace(old, new))
        self.mutated("execution_contract.json", lambda text: text.replace(
            str(m.STAGING_DIRECTORY), str(m.RAW_DIRECTORY)))
        with self.assertRaises(m.Failure):
            m.validate(m.HW / "dc_handoff/runs")

    def test_remaining_old_guards_reject(self):
        self.mutated("lm_discovery.rc", lambda text: "1\n")
        self.mutated("lm_discovery.log", lambda text: text.replace("registered=0 value_hex=", "registered=1 value_hex=", 1))
        self.mutated("lm_discovery.log", lambda text: text.replace("exact=1 value_hex=", "exact=0 value_hex=", 1))
        self.mutated("lm_discovery.log", lambda text: text.replace("frame_files=0 ndm_files=0", "frame_files=1 ndm_files=0"))
        self.mutated("same_uid_census_after.json", lambda text: text.replace('"PASS_EMPTY"', '"CHANGED"'))
        self.mutated("repo_root_after.json", lambda text: '{"changed": true}')
        self.mutated("execution_output_manifest.json", lambda text: text.replace('"lm_return_code": 0', '"lm_return_code": 1'))


if __name__ == "__main__":
    unittest.main()
