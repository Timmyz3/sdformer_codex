import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / (
    "hw_autoresearch_nts07/dc_handoff/scripts/"
    "seal_formality_snapshot_r2.sh"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SealFormalitySnapshotR2HardeningTest(unittest.TestCase):
    def make_fixture(self, base, filelist_text="rtl/design.sv\n",
                     duplicate_live_alias=False):
        base = Path(base)
        hw = base / "hw"
        run = base / "run"
        (hw / "rtl").mkdir(parents=True)
        (hw / "filelists").mkdir()
        (run / "netlist").mkdir(parents=True)
        (run / "reports").mkdir()
        (hw / "rtl/design.sv").write_text("module design; endmodule\n")
        filelist = hw / "filelists/design.f"
        filelist.write_text(filelist_text)
        netlist = run / "netlist/design_mapped.v"
        netlist.write_text("module design; endmodule\n")
        svf = run / "netlist/design.svf"
        svf.write_text("synthetic svf\n")
        library = base / "library.db"
        library.write_text("external library identity\n")
        output = run / "formality_attempt.log"
        output.write_text("Verification SUCCEEDED\n")
        manifest = {
            "mode": "formality",
            "design_name": "design",
            "paths": {
                "LIB_DB": {"path": str(library), "sha256": digest(library)},
                "RTL_FILELIST": {
                    "path": str(filelist), "sha256": digest(filelist)},
                "MAPPED_NETLIST": {
                    "path": str(netlist), "sha256": digest(netlist)},
            },
        }
        (run / "formality_run_manifest.json").write_text(json.dumps(manifest))
        ledger_lines = ["{}  {}\n".format(digest(output), output)]
        if duplicate_live_alias:
            (run / "sub").mkdir()
            ledger_lines.append("{}  {}\n".format(
                digest(output), run / "sub/../formality_attempt.log"
            ))
        (run / "formality_evidence_attempt.sha256").write_text(
            "".join(ledger_lines)
        )
        return hw, run

    def invoke(self, hw, run, tag):
        environment = dict(os.environ)
        environment["PYTHON_BIN"] = "/usr/bin/python3.6"
        return subprocess.run(
            ["bash", str(SCRIPT), str(run), "attempt", tag, str(hw)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=environment,
        )

    def test_positive_snapshot_is_self_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            hw, run = self.make_fixture(directory)
            result = self.invoke(hw, run, "positive")
            self.assertEqual(result.returncode, 0, result.stdout)
            snapshot = run / "sealed_formality_positive"
            ledger = run / "sealed_formality_evidence_positive.sha256"
            self.assertTrue(snapshot.is_dir())
            self.assertTrue(ledger.is_file())
            self.assertTrue((snapshot / "formality_run_manifest.json").is_file())
            self.assertTrue((snapshot / "formality_live_evidence.sha256").is_file())

    def test_filelist_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "outside.sv"
            outside.write_text("module outside; endmodule\n")
            hw, run = self.make_fixture(directory, "../outside.sv\n")
            result = self.invoke(hw, run, "escape")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("escapes canonical root", result.stdout)
            self.assertFalse((run / "sealed_formality_escape").exists())

    def test_symlink_source_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outside = base / "outside.sv"
            outside.write_text("module outside; endmodule\n")
            hw, run = self.make_fixture(directory, "rtl/link.sv\n")
            (hw / "rtl/link.sv").symlink_to(outside)
            result = self.invoke(hw, run, "symlink_escape")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("escapes canonical root", result.stdout)

    def test_same_byte_normalized_target_collision_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            hw, run = self.make_fixture(directory, duplicate_live_alias=True)
            result = self.invoke(hw, run, "collision")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("target collision", result.stdout)
            self.assertFalse((run / "sealed_formality_collision").exists())


if __name__ == "__main__":
    unittest.main()
