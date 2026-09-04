import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "system_simulator/scripts/canonicalize_m2133_icc2_corner_spef.py"
SPEC = importlib.util.spec_from_file_location("m2133_spef", SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class M2133SpefCanonicalizerTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        raw = root / "raw_parasitics"
        output = root / "output"
        raw.mkdir()
        output.mkdir()
        return temporary, raw, output, root / "receipt.json"

    def test_unique_tt_25_corner_is_atomically_canonicalized(self):
        temporary, raw, output, receipt = self.fixture()
        with temporary:
            source = raw / "routed.n28_1p9m_6x1z1u_typ_25.spef"
            source.write_text("*SPEF fixture\n")
            (raw / "routed.spef_scenario").write_text("tt_power fixture\n")
            result = MOD.canonicalize(raw, output, receipt)
            self.assertEqual(result["temperature_c"], 25.0)
            self.assertFalse(source.exists())
            self.assertTrue((output / "routed.spef").is_file())

    def test_no_spef_is_rejected(self):
        temporary, raw, output, receipt = self.fixture()
        with temporary, self.assertRaisesRegex(ValueError, "exactly one"):
            MOD.canonicalize(raw, output, receipt)

    def test_scenario_only_is_rejected(self):
        temporary, raw, output, receipt = self.fixture()
        with temporary:
            (raw / "routed.spef_scenario").write_text("metadata only\n")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                MOD.canonicalize(raw, output, receipt)

    def test_multiple_corner_spefs_are_rejected(self):
        temporary, raw, output, receipt = self.fixture()
        with temporary:
            (raw / "routed.n28_1p9m_6x1z1u_typ_25.spef").write_text("tt\n")
            (raw / "routed.n28_1p9m_6x1z1u_typ_125.spef").write_text("ss\n")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                MOD.canonicalize(raw, output, receipt)

    def test_wrong_corner_or_temperature_is_rejected(self):
        for name in ("routed.n28_1p9m_6x1z1u_typ_125.spef",
                     "routed.wrong_25.spef"):
            with self.subTest(name=name):
                temporary, raw, output, receipt = self.fixture()
                with temporary:
                    (raw / name).write_text("wrong\n")
                    with self.assertRaisesRegex(ValueError, "wrong raw parasitic"):
                        MOD.canonicalize(raw, output, receipt)

    def test_empty_or_symlink_spef_is_rejected(self):
        temporary, raw, output, receipt = self.fixture()
        with temporary:
            (raw / "routed.n28_1p9m_6x1z1u_typ_25.spef").write_text("")
            with self.assertRaisesRegex(ValueError, "nonempty regular"):
                MOD.canonicalize(raw, output, receipt)
        temporary, raw, output, receipt = self.fixture()
        with temporary:
            target = raw / "target"
            target.write_text("fixture\n")
            (raw / "routed.n28_1p9m_6x1z1u_typ_25.spef").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                MOD.canonicalize(raw, output, receipt)

    def test_preexisting_canonical_is_rejected(self):
        temporary, raw, output, receipt = self.fixture()
        with temporary:
            (raw / "routed.n28_1p9m_6x1z1u_typ_25.0.spef").write_text("tt\n")
            (output / "routed.spef").write_text("stale\n")
            with self.assertRaisesRegex(ValueError, "existed before"):
                MOD.canonicalize(raw, output, receipt)


if __name__ == "__main__":
    unittest.main()
