import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "hw_autoresearch_nts07/dc_handoff/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "build_m31_r5_synopsys_receipt.py"
SPEC = importlib.util.spec_from_file_location("m31r5receipt", str(SCRIPT))
R5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R5)
WORK_ROOT = Path("/home/zhumd/work")
RUNS_ROOT = WORK_ROOT / (
    "synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs")
RUN = RUNS_ROOT / R5.RUN_NAME
REANCHOR = WORK_ROOT / R5.REANCHOR_RELATIVE
RECEIPT = ROOT / R5.R5_RECEIPT_RELATIVE
RECEIPT_SHA256 = (
    "5135fb099e3bd434f928d5345c74e01db13c89604ae3956d1117782938b071b7")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_ledger(path, base, relatives):
    rows = []
    for relative in relatives:
        target = Path(base) / relative
        rows.append("{}  {}\n".format(digest(target), relative))
    Path(path).write_text("".join(rows), encoding="utf-8")


def clock_report(rows, footer):
    separator = "-" * 80
    return (
        "Report : clocks\n"
        "Clock          Period   Waveform            Attrs     Sources\n"
        + separator + "\n" + "\n".join(rows) + "\n"
        + separator + "\n" + str(footer) + "\n")


class BuildM31R5SynopsysReceiptTest(unittest.TestCase):
    def test_relative_exact_ledger_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_text("a\n")
            (root / "b").write_text("b\n")
            ledger = root / "ledger.sha256"
            write_ledger(ledger, root, ["a", "b"])
            _, rows = R5.parse_exact_relative_ledger(
                ledger, root, {"a", "b"}, "synthetic")
            self.assertEqual(set(rows), {"a", "b"})

    def test_rehashed_extra_ledger_entry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b", "extra"):
                (root / name).write_text(name + "\n")
            ledger = root / "ledger.sha256"
            write_ledger(ledger, root, ["a", "b", "extra"])
            with self.assertRaisesRegex(ValueError, "exact expected set"):
                R5.parse_exact_relative_ledger(
                    ledger, root, {"a", "b"}, "synthetic")

    def test_absolute_ledger_entry_rejected_even_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "a"
            target.write_text("a\n")
            ledger = root / "ledger.sha256"
            ledger.write_text("{}  {}\n".format(digest(target), target))
            with self.assertRaisesRegex(ValueError, "absolute/path-escape"):
                R5.parse_exact_relative_ledger(
                    ledger, root, {str(target)}, "synthetic")

    def test_symlink_path_replacement_rejected_even_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "real"
            target.write_text("same bytes\n")
            link = root / "link"
            link.symlink_to(target)
            ledger = root / "ledger.sha256"
            ledger.write_text("{}  link\n".format(digest(target)))
            with self.assertRaisesRegex(ValueError, "symlink"):
                R5.parse_exact_relative_ledger(
                    ledger, root, {"link"}, "synthetic")

    def test_out_of_root_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            root.mkdir()
            outside = base / "outside"
            outside.write_text("outside\n")
            with self.assertRaisesRegex(ValueError, "escapes"):
                R5.canonical_file(outside, root, "outside")

    def test_alternate_copied_work_root_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_work = Path(directory)
            with self.assertRaisesRegex(ValueError, "escapes"):
                R5.validate_roots(
                    fake_work, fake_work / "sdformer_codex/SDformer",
                    fake_work / (
                        "synopsys_date_dual/hw_autoresearch_nts07/"
                        "dc_handoff/runs"),
                    fake_work / (
                        "synopsys_date_dual/hw_autoresearch_nts07/"
                        "dc_handoff/runs") / R5.RUN_NAME)

    def test_normalized_ledger_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "a"
            target.write_text("a\n")
            ledger = root / "ledger.sha256"
            ledger.write_text("{}  ./a\n".format(digest(target)))
            with self.assertRaisesRegex(ValueError, "normalized/duplicate"):
                R5.parse_exact_relative_ledger(
                    ledger, root, {"a"}, "synthetic")

    def test_unique_clock_table_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "clocks.rpt"
            report.write_text(clock_report([
                "core_clk         3.00   {0 1.5}             f         {clk_core}",
            ], 1))
            self.assertEqual(R5.parse_unique_clock_report(report)[
                "clock_count"], 1)

    def test_rehashed_second_clock_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "clocks.rpt"
            report.write_text(clock_report([
                "core_clk         3.00   {0 1.5}             f         {clk_core}",
                "forged_clk       3.00   {0 1.5}             f         {forged}",
            ], 2))
            with self.assertRaisesRegex(ValueError, "clock table population"):
                R5.parse_unique_clock_report(report)

    def make_cross_binding(self, directory):
        base = Path(directory)
        fm = base / "fm"
        vcs = base / "vcs"
        dc = base / "dc"
        core = ROOT / (
            "hw_autoresearch_nts07/rtl_m31/"
            "qfit_atlif_unified_t10_t2_stream_core.sv")
        pool = ROOT / (
            "hw_autoresearch_nts07/rtl_m31/qfit_signed_int8_mul96_pool.sv")
        bindings = {
            "rtl_m31/qfit_signed_int8_mul96_pool.sv": pool,
            "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv": core,
        }
        for relative, source in bindings.items():
            for root, prefix in ((fm, "inputs/hw_root"),
                                 (vcs, "inputs/hw_root"),
                                 (dc, "inputs")):
                target = root / prefix / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        filelist = fm / (
            "inputs/hw_root/dc_handoff/filelists/"
            "date_m31_unified_t10_t2_dc.f")
        filelist.parent.mkdir(parents=True)
        filelist.write_text(
            "rtl_m31/qfit_signed_int8_mul96_pool.sv\n"
            "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv\n")
        manifest = vcs / "input_sha256.txt"
        other_inputs = {
            "verif_m31/qfit_atlif_unified_t10_t2_stream_assertions.sv":
                "assertions\n",
            "tb_m31/tb_qfit_atlif_unified_t10_t2_stream_core.sv": "tb\n",
            "dc_handoff/filelists/date_m31_unified_t10_t2_vcs.f":
                "filelist\n",
            "dc_handoff/scripts/run_vcs_m31_unified_t10_t2_sva.sh": "run\n",
        }
        manifest_lines = [
            "{}  rtl_m31/qfit_signed_int8_mul96_pool.sv\n".format(
                R5.POOL_SHA256),
            "{}  rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv\n".format(
                R5.CORE_SHA256),
        ]
        for relative, contents in other_inputs.items():
            target = vcs / "inputs/hw_root" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
            manifest_lines.append("{}  {}\n".format(digest(target), relative))
        manifest.write_text("".join(manifest_lines))
        return fm, vcs, dc

    def test_fm_vcs_dc_rtl_cross_binding_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            fm, vcs, dc = self.make_cross_binding(directory)
            result = R5.validate_exact_rtl_cross_binding(fm, vcs, dc)
            self.assertEqual(result[
                "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv"],
                R5.CORE_SHA256)

    def test_coherently_rehashed_alternate_rtl_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fm, vcs, dc = self.make_cross_binding(directory)
            alternate = vcs / (
                "inputs/hw_root/rtl_m31/"
                "qfit_atlif_unified_t10_t2_stream_core.sv")
            alternate.write_text("module forged; endmodule\n")
            manifest = vcs / "input_sha256.txt"
            manifest.write_text(manifest.read_text().replace(
                R5.CORE_SHA256, digest(alternate)))
            with self.assertRaisesRegex(ValueError, "cross-binding"):
                R5.validate_exact_rtl_cross_binding(fm, vcs, dc)

    def test_receipt_output_path_is_exact_and_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "receipt.json"
            with self.assertRaisesRegex(ValueError, "exact path"):
                R5.write_output(wrong, {"test": True}, ROOT)

    def test_receipt_output_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_repo = Path(directory) / "repo"
            fake_repo.mkdir()
            # Mock only the canonical repo admission so the output-parent check
            # is isolated without attempting to write into the real contract.
            linked = fake_repo / "hw_autoresearch_nts07/contracts"
            linked.parent.mkdir(parents=True)
            target = Path(directory) / "outside"
            target.mkdir()
            linked.symlink_to(target, target_is_directory=True)
            output = fake_repo / R5.R5_RECEIPT_RELATIVE
            with mock.patch.object(R5, "canonical_dir", side_effect=[
                    fake_repo, ValueError("contains a symlink")]):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    R5.write_output(output, {"test": True}, fake_repo)

    def test_live_r5_build_positive(self):
        args = argparse.Namespace(
            work_root=WORK_ROOT, repo_root=ROOT, runs_root=RUNS_ROOT,
            run_dir=RUN, reanchor_ledger=REANCHOR, date="2026-08-22",
            output=None)
        result = R5.build(args)
        self.assertEqual(result["schema"], R5.SCHEMA)
        self.assertFalse(result["advances"]["dc_or_formality_rerun"])
        self.assertEqual(result["frozen_dc_sta"][
            "unique_clock_contract"]["clock_count"], 1)
        self.assertFalse(result["headline_admitted"])

    def test_live_receipt_is_exact_deterministic_build(self):
        args = argparse.Namespace(
            work_root=WORK_ROOT, repo_root=ROOT, runs_root=RUNS_ROOT,
            run_dir=RUN, reanchor_ledger=REANCHOR, date="2026-08-22",
            output=None)
        expected = json.dumps(
            R5.build(args), indent=2, sort_keys=True) + "\n"
        self.assertEqual(RECEIPT.read_text(encoding="utf-8"), expected)
        self.assertEqual(digest(RECEIPT), RECEIPT_SHA256)

    def test_live_receipt_create_only_policy_rejects_overwrite(self):
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            R5.write_output(RECEIPT, {"forged": True}, ROOT)


if __name__ == "__main__":
    unittest.main()
