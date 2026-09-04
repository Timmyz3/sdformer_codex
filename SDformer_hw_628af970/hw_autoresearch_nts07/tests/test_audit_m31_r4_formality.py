import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / (
    "hw_autoresearch_nts07/dc_handoff/scripts/"
    "audit_m31_r4_formality.py"
)
SPEC = importlib.util.spec_from_file_location("m31fmaudit", str(SCRIPT))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
SEALER = ROOT / (
    "hw_autoresearch_nts07/dc_handoff/scripts/seal_formality_evidence.sh"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditM31R4FormalityTest(unittest.TestCase):
    def make_run(self, directory, extra_log=""):
        run = Path(directory) / "run"
        (run / "reports").mkdir(parents=True)
        (run / "netlist").mkdir()
        attempt = "fresh"
        log = run / ("formality_{}.log".format(attempt))
        log.write_text(
            "Verification SUCCEEDED\n"
            " 100 Passing compare points\n"
            "Failing (not equivalent) 0 0 0 0 0 0 0 0\n"
            " 0(0) Unmatched reference(implementation) compare points\n"
            " 0(0) Unmatched reference(implementation) primary inputs, black-box outputs\n"
            " 174(0) Unmatched reference(implementation) unread points\n"
            + extra_log
        )
        (run / ("formality_{}.exit_status".format(attempt))).write_text("0\n")
        (run / "reports/formality_status.txt").write_text("PASS\n")
        (run / "reports/formality_unmatched.rpt").write_text(
            "Report : unmatched_points\nNo unmatched points.\n")
        (run / "reports/formality_verify.rpt").write_text(
            "Report : failing_points\nNo failing compare points.\n")
        library = Path(directory) / "library.db"
        filelist = Path(directory) / "rtl.f"
        netlist = run / "netlist/qfit_atlif_unified_t10_t2_stream_core_mapped.v"
        library.write_text("library\n")
        filelist.write_text("rtl.sv\n")
        netlist.write_text("module m; endmodule\n")
        manifest = {
            "mode": "formality", "design_name": AUDIT.DESIGN,
            "paths": {
                "LIB_DB": {"path": str(library), "sha256": digest(library)},
                "RTL_FILELIST": {"path": str(filelist), "sha256": digest(filelist)},
                "MAPPED_NETLIST": {"path": str(netlist), "sha256": digest(netlist)},
            },
        }
        (run / "formality_run_manifest.json").write_text(json.dumps(manifest))
        return run, attempt

    def test_positive_records_unread_reference_and_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            run, attempt = self.make_run(directory)
            result = AUDIT.build(run, attempt, 100)
            verification = result["verification"]
            self.assertEqual(verification["unread_reference_points"], 174)
            self.assertEqual(verification["unread_implementation_points"], 0)
            self.assertEqual(verification["fmr_elab_147_diagnostics"], 0)

    def test_fmr_elab_147_diagnostic_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, attempt = self.make_run(
                directory,
                "Warning: index problem (FMR_ELAB-147)\n"
                "1 FMR_ELAB-147 messages produced\n",
            )
            with self.assertRaisesRegex(ValueError, "FMR_ELAB-147"):
                AUDIT.build(run, attempt, 100)

    def test_logic_simulator_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, attempt = self.make_run(
                directory,
                "Verification results may disagree with a logic simulator.\n",
            )
            with self.assertRaisesRegex(ValueError, "disagreement"):
                AUDIT.build(run, attempt, 100)

    def test_inconsistent_unread_population_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, attempt = self.make_run(
                directory,
                " 175(0) Unmatched reference(implementation) unread points\n",
            )
            with self.assertRaisesRegex(ValueError, "unread points population"):
                AUDIT.build(run, attempt, 100)

    def test_live_sealer_runs_strict_machine_audit_and_hashes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            run, attempt = self.make_run(directory)
            environment = dict(os.environ)
            environment["PYTHON_BIN"] = "/usr/bin/python3.6"
            result = subprocess.run(
                ["bash", str(SEALER), str(run), attempt, "100"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            machine = run / ("formality_machine_audit_{}.json".format(attempt))
            ledger = run / ("formality_evidence_{}.sha256".format(attempt))
            admission = run / ("formality_admission_{}.txt".format(attempt))
            self.assertTrue(machine.is_file())
            self.assertIn(str(machine), ledger.read_text(encoding="utf-8"))
            self.assertIn(
                "build_m31_r4_synopsys_receipt.py",
                ledger.read_text(encoding="utf-8"),
            )
            text = admission.read_text(encoding="utf-8")
            self.assertIn("unread_reference_points=174", text)
            self.assertIn("fmr_elab_147_diagnostics=0", text)


if __name__ == "__main__":
    unittest.main()
