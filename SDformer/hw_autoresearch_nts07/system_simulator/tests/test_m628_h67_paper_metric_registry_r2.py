#!/usr/bin/env python3

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/scripts/build_m628_h67_paper_metric_registry_r2.py"
CONFIG = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/config/m628_h67_paper_metric_registry_r2_20260828.json"
SPEC = importlib.util.spec_from_file_location("m628_registry", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class M628RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads(CONFIG.read_text(encoding="utf-8"))

    @staticmethod
    def remove_file(path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def write_config(self, obj):
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
        with handle:
            json.dump(obj, handle, ensure_ascii=False, allow_nan=False)
        self.addCleanup(self.remove_file, Path(handle.name))
        return Path(handle.name)

    def test_canonical_registry_passes_but_headline_fails_closed(self):
        result = MODULE.build(CONFIG)
        self.assertEqual(12, len(result["source_hashes_validated"]))
        self.assertEqual(0, result["headline_gate"]["eligible_row_count"])
        self.assertFalse(result["headline_gate"]["admitted"])
        self.assertFalse(result["analytical_diagnostic"]["admitted"])
        self.assertEqual("1.7942940217026179000564835389", result["analytical_diagnostic"]["speedup_low"])
        self.assertEqual("1.8234548159105851413543721236", result["analytical_diagnostic"]["speedup_high"])

    def test_source_sha_mutation_is_rejected(self):
        obj = copy.deepcopy(self.base)
        obj["sources"]["m528"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(MODULE.RegistryError, "source SHA mismatch"):
            MODULE.build(self.write_config(obj))

    def test_table_b_cannot_be_promoted(self):
        obj = copy.deepcopy(self.base)
        obj["table_b_schema"]["rows"][0]["headline_eligible"] = True
        with self.assertRaisesRegex(MODULE.RegistryError, "cannot be headline eligible"):
            MODULE.build(self.write_config(obj))

    def test_table_c_cannot_be_labelled_ours(self):
        obj = copy.deepcopy(self.base)
        obj["table_c_schema"]["rows"][0]["ours"] = True
        with self.assertRaisesRegex(MODULE.RegistryError, "cannot be labelled ours"):
            MODULE.build(self.write_config(obj))

    def test_analytical_range_cannot_be_admitted(self):
        obj = copy.deepcopy(self.base)
        obj["analytical_diagnostic"]["admitted"] = True
        with self.assertRaisesRegex(MODULE.RegistryError, "explicitly non-admitted"):
            MODULE.build(self.write_config(obj))

    def test_analytical_anchor_tamper_is_rejected(self):
        obj = copy.deepcopy(self.base)
        obj["analytical_diagnostic"]["expected_speedup_low"] = "1.80"
        with self.assertRaisesRegex(MODULE.RegistryError, "do not independently recompute"):
            MODULE.build(self.write_config(obj))

    def test_required_table_a_row_cannot_disappear(self):
        obj = copy.deepcopy(self.base)
        obj["table_a_schema"]["rows"] = [
            row for row in obj["table_a_schema"]["rows"] if row["row_id"] != "exact_bit_k1x8"
        ]
        with self.assertRaisesRegex(MODULE.RegistryError, "required row is missing"):
            MODULE.build(self.write_config(obj))

    def test_claimed_table_a_count_must_match_gate(self):
        obj = copy.deepcopy(self.base)
        obj["claim_boundary"]["table_a_admitted_rows"] = 1
        with self.assertRaisesRegex(MODULE.RegistryError, "disagrees with executable gate"):
            MODULE.build(self.write_config(obj))

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as handle:
            handle.write('{"schema":"x","schema":"y"}')
            path = Path(handle.name)
        self.addCleanup(self.remove_file, path)
        with self.assertRaisesRegex(MODULE.RegistryError, "duplicate JSON key"):
            MODULE.load_json(path)

    def test_nonfinite_json_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as handle:
            handle.write('{"value":NaN}')
            path = Path(handle.name)
        self.addCleanup(self.remove_file, path)
        with self.assertRaisesRegex(MODULE.RegistryError, "non-finite JSON number"):
            MODULE.load_json(path)

    def test_symlink_source_component_is_rejected(self):
        tests_dir = REPO_ROOT / "hw_autoresearch_nts07/system_simulator/tests"
        with tempfile.TemporaryDirectory(dir=tests_dir) as temp_dir:
            root = Path(temp_dir)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)
            relative = link.relative_to(REPO_ROOT).as_posix()
            with self.assertRaisesRegex(MODULE.RegistryError, "symlink source component refused"):
                MODULE.secure_repo_file(relative)


if __name__ == "__main__":
    unittest.main()
