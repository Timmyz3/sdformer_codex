from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "build_m1354_ep34_table_a_lexical_path_authority_compiler.py")
M1351_TEST = ROOT / (
    "hw_autoresearch_nts07/system_simulator/tests/"
    "test_m1351_ep34_table_a_memory_timed_authority_compiler.py")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec failed: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


N = load("m1354_source", SOURCE)
T = load("m1354_bound_m1351_test", M1351_TEST)


class Tests(unittest.TestCase):
    def test_01_legal_regular_leaf_and_missing_output_leaf(self):
        with tempfile.TemporaryDirectory(prefix="m1354_regular_") as temporary:
            root = Path(temporary)
            regular = root / "config.json"
            regular.write_text("{}\n"); regular.chmod(0o444)
            self.assertEqual(
                N.lexical_lstat_then_resolved_containment(root, regular), regular)
            output = root / "future.json"
            self.assertEqual(N.lexical_lstat_then_resolved_containment(
                root, output, leaf_must_exist=False), output)

    def test_02_symlink_leaf_is_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory(prefix="m1354_leaf_") as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}\n"); target.chmod(0o444)
            alias = root / "alias.json"; alias.symlink_to(target.name)
            with self.assertRaisesRegex(N.CompileError, "symlink lexical component"):
                N.lexical_lstat_then_resolved_containment(root, alias)

    def test_03_symlink_ancestor_is_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory(prefix="m1354_ancestor_") as temporary:
            root = Path(temporary)
            real = root / "real"; real.mkdir()
            leaf = real / "config.json"; leaf.write_text("{}\n"); leaf.chmod(0o444)
            alias = root / "alias"; alias.symlink_to(real.name, target_is_directory=True)
            with self.assertRaisesRegex(N.CompileError, "symlink lexical component"):
                N.lexical_lstat_then_resolved_containment(root, alias / leaf.name)

    def test_04_broken_symlink_leaf_is_rejected_as_symlink(self):
        with tempfile.TemporaryDirectory(prefix="m1354_broken_") as temporary:
            root = Path(temporary)
            broken = root / "broken.json"; broken.symlink_to("missing.json")
            with self.assertRaisesRegex(N.CompileError, "symlink lexical component"):
                N.lexical_lstat_then_resolved_containment(root, broken)

    def test_05_m1353_exact_accepted_attack_now_fails_build(self):
        fixture = T.F.Fixture()
        self.addCleanup(fixture.close)
        T.add_m1351_authorities(fixture)
        real = fixture.config_path()
        alias = fixture.root / "config_alias.json"; alias.symlink_to(real.name)
        with self.assertRaisesRegex(N.CompileError, "symlink lexical component"):
            N.build(alias, fixture.root, fixture.allowlist)

    def test_06_regular_fixture_build_stays_source_only(self):
        fixture = T.F.Fixture()
        self.addCleanup(fixture.close)
        T.add_m1351_authorities(fixture)
        result = N.build(fixture.config_path(), fixture.root, fixture.allowlist)
        self.assertEqual(result["status"],
                         "PASS_SOURCE_FIXTURE_LEXICAL_PATH_NOT_PRODUCTION")
        self.assertEqual(result["claim_boundary"]["production_rows"], 0)
        self.assertFalse(result["claim_boundary"]["paper_headline_admitted"])
        self.assertEqual(N.M1351.M1342.PRODUCTION_AUTHORITY_ALLOWLIST, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
