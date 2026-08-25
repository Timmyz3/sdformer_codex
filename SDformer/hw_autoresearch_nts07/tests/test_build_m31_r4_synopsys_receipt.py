import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "hw_autoresearch_nts07/dc_handoff/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "build_m31_r4_synopsys_receipt.py"
SPEC = importlib.util.spec_from_file_location("m31receipt", str(SCRIPT))
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
FUNCTIONAL_RECEIPT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m31_output_receipt_r4_static_phase_20260822.json")
FUNCTIONAL_ADMISSION = ROOT / (
    "hw_autoresearch_nts07/results/"
    "m31_r4_static_phase_vcs_machine_admission_20260822/"
    "m31_r4_static_phase_vcs_machine_admission.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def write_ledger(path, entries, base=None):
    base = Path(base).resolve() if base is not None else None
    rows = []
    for entry in entries:
        entry = Path(entry).resolve()
        name = str(entry.relative_to(base)) if base is not None else str(entry)
        rows.append("{}  {}\n".format(digest(entry), name))
    Path(path).write_text("".join(rows), encoding="utf-8")


class BuildM31R4SynopsysReceiptTest(unittest.TestCase):
    def make_fixture(self, directory):
        base = Path(directory)
        run = base / "run"
        reports = run / "reports"
        netlist = run / "netlist"
        reports.mkdir(parents=True)
        netlist.mkdir()
        attempt = "fresh_r4"
        snapshot_tag = "fresh_r4_snapshot"

        (reports / "qor.rpt").write_text(
            "Design : qfit_atlif_unified_t10_t2_stream_core\n"
            "  Hierarchical Cell Count: 97\n"
            "  Leaf Cell Count: 96\n"
            "  Macro Count: 0\n"
            "  Net Area: 0.000000\n")
        (reports / "area.rpt").write_text(
            "Number of cells: 193\n"
            "Number of combinational cells: 96\n"
            "Number of sequential cells: 0\n"
            "Number of macros/black boxes: 0\n"
            "Net Interconnect area: undefined  (Wire load has zero net area)\n"
            "Total cell area: 100.000000\n")
        (reports / "clocks.rpt").write_text(
            "core_clk 3.00 {0 1.5} f {clk_core}\n")
        resource = [
            "stage=postcompile", "pool_count=1", "leaf_count=96",
            "pool_path=u_mul_pool",
        ]
        resource.extend(
            "leaf=u_mul_pool/u{} ref=qfit_signed_int8_mul_leaf_{} "
            "mapped_cells=1 mapped_area=1.0".format(index, index)
            for index in range(96))
        resource.extend([
            "pool_external_leaf_count=0", "empty_mapped_leaf_count=0",
            "status=PASS_EXACT_ONE_POOL_96_LEAVES",
        ])
        (reports / "m31_resource_audit_postcompile.rpt").write_text(
            "\n".join(resource) + "\n")
        (reports / "references_postcompile.rpt").write_text("mapped cells\n")
        (reports / "timing_setup.rpt").write_text(" slack (MET) 0.0010\n")
        (reports / "timing_hold.rpt").write_text(" slack (MET) 0.0020\n")
        (run / "dc.log").write_text("DC complete\n")
        (run / "admission.txt").write_text(
            "status=PASS_EXACT96_PREMACRO_LOGIC_ONLY\n")
        mapped = netlist / (BUILDER.DESIGN + "_mapped.v")
        svf = netlist / (BUILDER.DESIGN + ".svf")
        mapped.write_text("module m; endmodule\n")
        svf.write_text("svf\n")

        dc_audit = reports / "m31_r4_dc_machine_audit.json"
        write_json(dc_audit, BUILDER.rebuild_dc_audit(run, 3.000))
        dc_live_required = [
            dc_audit, run / "admission.txt", run / "dc.log",
            reports / "m31_resource_audit_postcompile.rpt",
            reports / "qor.rpt", reports / "area.rpt",
            reports / "clocks.rpt", reports / "references_postcompile.rpt",
            reports / "timing_setup.rpt", reports / "timing_hold.rpt",
            mapped, svf,
        ]
        dc_evidence = run / "evidence.sha256"
        write_ledger(dc_evidence, dc_live_required)
        sealed_dc = run / "sealed_dc"
        sealed_dc.mkdir()
        sealed_marker = sealed_dc / "source_map.tsv"
        sealed_marker.write_text("sealed\n")
        sealed_dc_ledger = run / "sealed_dc_evidence.sha256"
        write_ledger(sealed_dc_ledger,
                     [sealed_marker, dc_evidence] + dc_live_required,
                     base=run)

        fm_log = run / "formality_{}.log".format(attempt)
        fm_exit = run / "formality_{}.exit_status".format(attempt)
        fm_status = reports / "formality_status.txt"
        fm_unmatched = reports / "formality_unmatched.rpt"
        fm_verify = reports / "formality_verify.rpt"
        fm_log.write_text(
            "Verification SUCCEEDED\n"
            " 100 Passing compare points\n"
            "Failing (not equivalent) 0 0 0 0 0 0 0 0\n"
            " 0(0) Unmatched reference(implementation) compare points\n"
            " 0(0) Unmatched reference(implementation) primary inputs, black-box outputs\n"
            " 174(0) Unmatched reference(implementation) unread points\n")
        fm_exit.write_text("0\n")
        fm_status.write_text("PASS\n")
        fm_unmatched.write_text("No unmatched points.\n")
        fm_verify.write_text("No failing compare points.\n")
        library = base / "library.db"
        filelist = base / "date_m31_unified_t10_t2_dc.f"
        library.write_text("library\n")
        filelist.write_text(
            "rtl_m31/qfit_signed_int8_mul96_pool.sv\n"
            "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv\n")
        manifest = run / "formality_run_manifest.json"
        write_json(manifest, {
            "mode": "formality", "design_name": BUILDER.DESIGN,
            "paths": {
                "LIB_DB": {"path": str(library), "sha256": digest(library)},
                "RTL_FILELIST": {
                    "path": str(filelist), "sha256": digest(filelist)},
                "MAPPED_NETLIST": {
                    "path": str(mapped), "sha256": digest(mapped)},
            },
        })
        fm_audit = run / "formality_machine_audit_{}.json".format(attempt)
        write_json(fm_audit,
                   BUILDER.rebuild_formality_audit(run, attempt, 100))
        fm_admission = run / "formality_admission_{}.txt".format(attempt)
        fm_admission.write_text(
            "status=PASS_RTL_TO_MAPPED_NETLIST_FORMALITY\n")
        fm_required = [
            fm_audit, fm_admission, fm_log, fm_exit, fm_status,
            fm_unmatched, fm_verify, manifest, SCRIPT,
        ]
        fm_evidence = run / "formality_evidence_{}.sha256".format(attempt)
        write_ledger(fm_evidence, fm_required)

        snapshot = run / "sealed_formality_{}".format(snapshot_tag)
        required_snapshot = [
            snapshot / "external_identity.sha256",
            snapshot / "source_map.tsv",
            snapshot / "formality_live_evidence.sha256",
            snapshot / "formality_run_manifest.json",
            snapshot / "seal_formality_snapshot_r2.sh",
            snapshot / "inputs/run/netlist/{}_mapped.v".format(BUILDER.DESIGN),
            snapshot / "inputs/run/netlist/{}.svf".format(BUILDER.DESIGN),
            snapshot / "inputs/hw_root/rtl_m31/qfit_signed_int8_mul96_pool.sv",
            snapshot / "inputs/hw_root/rtl_m31/"
            "qfit_atlif_unified_t10_t2_stream_core.sv",
            snapshot / "inputs/hw_root/dc_handoff/filelists/"
            "date_m31_unified_t10_t2_dc.f",
            snapshot / "inputs/hw_root/dc_handoff/scripts/"
            "build_m31_r4_synopsys_receipt.py",
            snapshot / "outputs/formality_{}.log".format(attempt),
            snapshot / "outputs/formality_{}.exit_status".format(attempt),
            snapshot / "outputs/reports/formality_status.txt",
            snapshot / "outputs/reports/formality_unmatched.rpt",
            snapshot / "outputs/reports/formality_verify.rpt",
            snapshot / "outputs/formality_machine_audit_{}.json".format(attempt),
            snapshot / "outputs/formality_admission_{}.txt".format(attempt),
            snapshot / "outputs/formality_run_manifest.json",
        ]
        for index, path in enumerate(required_snapshot):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("snapshot artifact {}\n".format(index))
        snapshot_ledger = run / (
            "sealed_formality_evidence_{}.sha256".format(snapshot_tag))
        write_ledger(snapshot_ledger, required_snapshot, base=run)

        args = argparse.Namespace(
            run_dir=run, attempt=attempt, snapshot_tag=snapshot_tag,
            snapshot_ledger=snapshot_ledger, dc_audit=dc_audit,
            formality_audit=fm_audit,
            functional_receipt=FUNCTIONAL_RECEIPT,
            functional_admission=FUNCTIONAL_ADMISSION,
            independent_review_score=94, date="2026-08-22",
            output=base / "receipt.json")
        return args

    def test_positive_build_has_correct_cell_semantics_and_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(directory)
            result = BUILDER.build(args)
            self.assertEqual(result["schema"], BUILDER.RECEIPT_SCHEMA)
            cells = result["dc_sta"]["cell_accounting"]
            self.assertEqual(cells["total_cell_instances_including_hierarchy"],
                             193)
            self.assertEqual(cells["leaf_mapped_cell_instances"], 96)
            self.assertEqual(result["formality"][
                "fmr_elab_147_diagnostics"], 0)
            self.assertFalse(result["headline_admitted"])

    def test_rehashed_fmr_diagnostic_still_blocks_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(directory)
            log = Path(args.run_dir) / "formality_{}.log".format(args.attempt)
            log.write_text(log.read_text() +
                           "Warning: bad index (FMR_ELAB-147)\n")
            write_json(args.formality_audit,
                       json.loads(Path(args.formality_audit).read_text()))
            with self.assertRaisesRegex(ValueError, "FMR_ELAB-147"):
                BUILDER.build(args)

    def test_snapshot_ledger_must_cover_every_snapshot_file(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(directory)
            extra = (Path(args.run_dir)
                     / "sealed_formality_{}".format(args.snapshot_tag)
                     / "unlisted.txt")
            extra.write_text("not in ledger\n")
            with self.assertRaisesRegex(ValueError, "not exactly closed"):
                BUILDER.build(args)

    def test_missing_placeholder_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(directory)
            Path(args.dc_audit).unlink()
            with self.assertRaisesRegex(ValueError, "missing M31 receipt"):
                BUILDER.build(args)

    def test_noncanonical_snapshot_ledger_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(directory)
            copied = Path(directory) / "copied_snapshot.sha256"
            copied.write_bytes(Path(args.snapshot_ledger).read_bytes())
            args.snapshot_ledger = copied
            with self.assertRaisesRegex(ValueError, "canonical run-local"):
                BUILDER.build(args)

    def test_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_fixture(directory)
            result = BUILDER.build(args)
            BUILDER.write_output(args.output, result)
            original = Path(args.output).read_bytes()
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                BUILDER.write_output(args.output, result)
            self.assertEqual(Path(args.output).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
